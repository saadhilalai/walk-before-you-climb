"""
walker_env.py — builds the courses and judges every episode.

    BipedalWalker-v3 / BipedalWalkerHardcore-v3   the physics simulation (official reward)
      -> WalkerStats   notes how the episode ended (finished / fell / backed_off / timed_out),
                       how far the robot got, and which obstacle it was at when it fell

WalkerStats never changes the reward or what the robot sees, so the score is
always the official one.
"""

import Box2D
import gymnasium as gym
from gymnasium.envs.box2d import bipedal_walker as bw

import config

STEP = bw.TERRAIN_STEP                                   # width of one terrain tile
START_X = STEP * bw.TERRAIN_STARTPAD / 2                 # where the body starts (4.67)
FINISH_X = (bw.TERRAIN_LENGTH - bw.TERRAIN_GRASS) * STEP  # crossing this ends the episode (88.67)


def course_obstacles(game):
    """List the obstacles of the current course, in order, as dicts with
    kind ("stump" / "pit" / "stairs_up" / "stairs_down"), x_start and x_end.

    The simulator doesn't keep this list, but every obstacle type is built from
    blocks of one fixed shape (in tiles): stumps are 1x1 or 2x2 squares, pit walls
    are 1 wide and 4 deep (two per pit), stair steps are 4 wide and 1 high."""
    blocks = []
    for body in game.terrain:
        shape = body.fixtures[0].shape
        if not isinstance(shape, Box2D.b2PolygonShape):
            continue                                     # the ground line itself
        xs = [v[0] for v in shape.vertices]
        ys = [v[1] for v in shape.vertices]
        w, h = round((max(xs) - min(xs)) / STEP), round((max(ys) - min(ys)) / STEP)
        blocks.append(dict(w=w, h=h, x0=min(xs), x1=max(xs), y=min(ys)))
    blocks.sort(key=lambda b: b["x0"])                   # the simulator stores them finish-to-start

    obstacles, i = [], 0
    while i < len(blocks):
        b = blocks[i]
        if b["w"] == b["h"]:                             # stump
            obstacles.append(dict(kind="stump", x_start=b["x0"], x_end=b["x1"]))
            i += 1
        elif (b["w"], b["h"]) == (1, 4):                 # pit: two walls, gap in between
            obstacles.append(dict(kind="pit", x_start=b["x0"], x_end=blocks[i + 1]["x1"]))
            i += 2
        elif (b["w"], b["h"]) == (4, 1):                 # staircase: touching steps
            j = i
            while j + 1 < len(blocks) and (blocks[j + 1]["w"], blocks[j + 1]["h"]) == (4, 1) \
                    and abs(blocks[j + 1]["x0"] - blocks[j]["x1"]) < 1e-3:
                j += 1
            kind = "stairs_up" if blocks[j]["y"] >= b["y"] else "stairs_down"
            obstacles.append(dict(kind=kind, x_start=b["x0"], x_end=blocks[j]["x1"]))
            i = j + 1
        else:
            raise ValueError(f"unknown terrain block {b}")
    return obstacles


def progress(x):
    """How far along the course x is: 0% at the start position, 100% at the finish line."""
    return min(100.0, max(0.0, 100 * (x - START_X) / (FINISH_X - START_X)))


class WalkerStats(gym.Wrapper):
    """Adds to the final step's `info`: outcome, distance (% of the course, furthest
    point the body reached), end_x (where the body ended up) and, on hardcore, the obstacle
    the robot fell at (see obstacle_near)."""

    def reset(self, **kwargs):
        # Start every episode in a brand-new physics world, exactly like a freshly created
        # simulator. The game normally reuses one Box2D world, and leftover internal
        # bookkeeping from earlier episodes slightly changes later ones (found in Step 2:
        # the same model on the same course scored differently depending on what ran before).
        # With a fresh world, course seed s + the same moves always gives the same result.
        game = self.env.unwrapped
        game._destroy()
        game.world = Box2D.b2World()
        obs, info = self.env.reset(**kwargs)
        self.max_x = game.hull.position.x
        self.obstacles = course_obstacles(game) if game.hardcore else []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped
        x = game.hull.position.x
        self.max_x = max(self.max_x, x)
        if terminated or truncated:
            if game.game_over:
                outcome = "fell"
            elif x < 0:
                outcome = "backed_off"
            elif x > FINISH_X:
                outcome = "finished"
            else:
                outcome = "timed_out"
            info["outcome"] = outcome
            info["distance"] = progress(self.max_x)
            info["end_x"] = float(x)
            info["n_obstacles"] = len(self.obstacles)
            info["fell_at"] = self.obstacle_near(x) if outcome == "fell" else None
        return obs, reward, terminated, truncated, info

    def obstacle_near(self, x):
        """The obstacle the robot fell at (config.FALL_WINDOW_TILES): its body centre was from
        2 tiles before the obstacle to 3 tiles after it. If two qualify, the nearer one (the
        earlier one on a tie). "open ground" if none. (Until Step 4 the window was 2 tiles
        before to 1 tile after, and the label was "flat ground".)"""
        before, after = config.FALL_WINDOW_TILES["before"] * STEP, config.FALL_WINDOW_TILES["after"] * STEP
        best = None
        for ob in self.obstacles:
            if ob["x_start"] - before <= x <= ob["x_end"] + after:
                gap = max(ob["x_start"] - x, x - ob["x_end"], 0.0)
                if best is None or gap < best[0]:
                    best = (gap, ob["kind"])
        if best:
            return best[1]
        return "open ground" if self.obstacles else None


def make_env(course="normal", render_mode=None):
    """One course with the official reward and the outcome judge on top."""
    c = config.COURSES[course]
    env = gym.make(c["env_id"], render_mode=render_mode)
    assert env.spec.max_episode_steps == c["time_limit"]
    return WalkerStats(env)
