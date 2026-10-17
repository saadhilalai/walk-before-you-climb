"""
render.py — draws the walker game sharply at any size, plus the fonts for labels.

The simulator only draws at 600x400, and its physics is tied to that drawing scale, so we
can't just turn the resolution up. Instead, render_hires() redraws the same physics objects
(sky, clouds, ground, obstacles, robot, lidar flash, start flag) as vector shapes at k times
the size: real detail, not blurry upscaling, with the same camera, colors and look as the game.
Drawing never touches the physics, so a replay with drawing gives exactly the same result.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # draw off-screen, no window

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from gymnasium.envs.box2d import bipedal_walker as bw  # noqa: E402
from PIL import ImageFont  # noqa: E402
from pygame import gfxdraw  # noqa: E402

FONTS = {False: ("/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "DejaVuSans.ttf"),
         True: ("/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "DejaVuSans-Bold.ttf")}
_font_cache = {}


def font(size, bold=False):
    """Helvetica Neue on a Mac (index 1 = bold), DejaVu Sans on Linux, Pillow's built-in font otherwise."""
    key = (size, bold)
    if key not in _font_cache:
        for path in FONTS[bold]:
            try:
                index = 1 if bold and path.endswith(".ttc") else 0
                _font_cache[key] = ImageFont.truetype(path, size, index=index)
                break
            except OSError:
                continue
        else:
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


def render_hires(game, k):
    """BipedalWalker's own render(), redrawn at k times the size. Returns an RGB array
    (400k x 600k x 3). Like the original, it advances the lidar-flash counter, which only
    affects drawing."""
    S = bw.SCALE * k
    W, H = round(bw.VIEWPORT_W * k), round(bw.VIEWPORT_H * k)
    view_w = bw.VIEWPORT_W / bw.SCALE              # 20 world units across
    sx = max(0.0, game.scroll)                     # the original crops its canvas the same way
    lw = max(1, round(k))

    def pt(x, y):
        return (x - sx) * S, H - y * S             # world (y up) -> screen (y down)

    surf = pygame.Surface((W, H))
    surf.fill((215, 215, 255))                     # sky

    for poly, x1, x2 in game.cloud_poly:           # clouds move at half speed (parallax)
        if x2 < game.scroll / 2 or x1 > game.scroll / 2 + view_w:
            continue
        pts = [((p[0] + game.scroll / 2 - sx) * S, H - p[1] * S) for p in poly]
        pygame.draw.polygon(surf, (255, 255, 255), pts)
        gfxdraw.aapolygon(surf, pts, (255, 255, 255))

    for poly, color in game.terrain_poly:          # the ground
        if poly[1][0] < game.scroll or poly[0][0] > game.scroll + view_w:
            continue
        pts = [pt(*c) for c in poly]
        pygame.draw.polygon(surf, color, pts)
        gfxdraw.aapolygon(surf, pts, color)

    game.lidar_render = (game.lidar_render + 1) % 100   # one lidar ray flashes now and then
    i = game.lidar_render
    if i < 2 * len(game.lidar):
        ray = game.lidar[i] if i < len(game.lidar) else game.lidar[len(game.lidar) - i - 1]
        if hasattr(ray, "p1") and hasattr(ray, "p2"):
            pygame.draw.line(surf, (255, 0, 0), pt(*ray.p1), pt(*ray.p2), lw)

    for obj in game.drawlist:                      # obstacles, ground edges, legs, body
        for f in obj.fixtures:
            trans = f.body.transform
            if type(f.shape) is bw.circleShape:
                c = pt(*(trans * f.shape.pos))
                pygame.draw.circle(surf, obj.color1, c, f.shape.radius * S)
                continue
            path = [pt(*(trans * v)) for v in f.shape.vertices]
            if len(path) > 2:
                pygame.draw.polygon(surf, obj.color1, path)
                gfxdraw.aapolygon(surf, path, obj.color1)
                pygame.draw.polygon(surf, obj.color2, path, width=lw)
                gfxdraw.aapolygon(surf, path, obj.color2)
            else:
                pygame.draw.line(surf, obj.color1, path[0], path[1], lw)

    x, y1 = pt(bw.TERRAIN_STEP * 3, bw.TERRAIN_HEIGHT)   # the start flag
    y2 = y1 - 50 * k
    pygame.draw.line(surf, (0, 0, 0), (x, y1), (x, y2), lw)
    flag = [(x, y2), (x, y2 + 10 * k), (x + 25 * k, y2 + 5 * k)]
    pygame.draw.polygon(surf, (230, 51, 0), flag)
    pygame.draw.lines(surf, (0, 0, 0), True, flag, lw)

    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)).copy()
