/* walker.js — Gymnasium 1.4.0's BipedalWalker-v3 / BipedalWalkerHardcore-v3, rewritten in
   JavaScript on planck.js (a JavaScript Box2D), plus walker_env.py's judge (finished / fell /
   backed off / timed out, and how far the robot got).

   Same constants, same course generator (NumPy's random numbers, see rng.js), same robot, same
   motors, same lidar, same observation and the same official score. Every number handed to the
   physics is rounded to 32 bits first, as Box2D stores it. The physics itself runs in 64-bit
   numbers (planck.js), Box2D's in 32-bit, so a long run can drift apart from Python's: how often
   that changes the outcome is measured by check_browser_sim.mjs: on the 600 hardcore test runs, only
   50.8% ended the same way, so the website replays the recorded runs instead. */

import * as planck from "./vendor/planck-1.5.0.mjs";
import { Generator } from "./rng.js";

const { Vec2 } = planck;
const f32 = Math.fround;

export const FPS = 50;
export const SCALE = 30.0;
const MOTORS_TORQUE = 80;
const SPEED_HIP = 4;
const SPEED_KNEE = 6;
const LIDAR_RANGE = 160 / SCALE;
const INITIAL_RANDOM = 5;
export const HULL_POLY = [[-30, 9], [6, 9], [34, 1], [34, -8], [-30, -8]];
const LEG_DOWN = -8 / SCALE;
export const LEG_W = 8 / SCALE, LEG_H = 34 / SCALE;
export const VIEWPORT_W = 600, VIEWPORT_H = 400;
export const TERRAIN_STEP = 14 / SCALE;
export const TERRAIN_LENGTH = 200;
export const TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4;
const TERRAIN_GRASS = 10;
export const TERRAIN_STARTPAD = 20;
const FRICTION = 2.5;
const GRASS = 0, STUMP = 1, STAIRS = 2, PIT = 3, N_STATES = 4;

export const START_X = TERRAIN_STEP * TERRAIN_STARTPAD / 2;
export const FINISH_X = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP;
export const TIME_LIMIT = { normal: 1600, hardcore: 2000 };

const v2 = (x, y) => Vec2(f32(x), f32(y));
const sign = x => (x > 0 ? 1 : x < 0 ? -1 : 0);

export function progress(x) {
  return Math.min(100, Math.max(0, 100 * (x - START_X) / (FINISH_X - START_X)));
}

export class Walker {
  /* new Walker(courseSeed, hardcore) builds the course and the robot exactly like
     env.reset(seed=courseSeed), including the zero-action first step that reset() takes. */
  constructor(courseSeed, hardcore = true, debug = {}) {
    this.hardcore = hardcore;
    this.timeLimit = hardcore ? TIME_LIMIT.hardcore : TIME_LIMIT.normal;
    this.rng = new Generator(courseSeed);
    this.world = new planck.World({ gravity: Vec2(0, -10), continuousPhysics: debug.continuous ?? true });
    this.gameOver = false;
    this.prevShaping = null;
    this.obstacles = [];          // [{poly, kind}] for drawing
    this._generateTerrain();
    this._generateClouds();
    this._createRobot();

    this.world.on("begin-contact", c => {
      const a = c.getFixtureA().getBody(), b = c.getFixtureB().getBody();
      if (a === this.hull || b === this.hull) this.gameOver = true;
      for (const leg of [this.legs[1], this.legs[3]]) if (a === leg || b === leg) leg.groundContact = true;
    });
    this.world.on("end-contact", c => {
      const a = c.getFixtureA().getBody(), b = c.getFixtureB().getBody();
      for (const leg of [this.legs[1], this.legs[3]]) if (a === leg || b === leg) leg.groundContact = false;
    });

    this.lidar = Array.from({ length: 10 }, () => ({ fraction: 1, p1: null, p2: null }));
    this.obs = this._physicsStep([0, 0, 0, 0]).obs;          // reset() ends with step([0, 0, 0, 0])
    this.steps = 0;
    this.score = 0;
    this.maxX = this.hull.getPosition().x;
    this.done = false;
    this.outcome = null;
  }

  _staticPoly(poly, opts = {}) {
    const body = this.world.createBody({ type: "static" });
    body.createFixture({ shape: planck.Polygon(poly.map(([x, y]) => v2(x, y))), friction: f32(FRICTION), ...opts });
    return body;
  }

  _generateTerrain() {
    const rng = this.rng, TS = TERRAIN_STEP;
    let state = GRASS, velocity = 0.0, y = TERRAIN_HEIGHT, counter = TERRAIN_STARTPAD, oneshot = false;
    let stairSteps = 0, stairWidth = 0, stairHeight = 0, originalY = 0;
    this.terrainX = [];
    this.terrainY = [];
    for (let i = 0; i < TERRAIN_LENGTH; i++) {
      const x = i * TS;
      this.terrainX.push(x);
      if (state === GRASS && !oneshot) {
        velocity = 0.8 * velocity + 0.01 * sign(TERRAIN_HEIGHT - y);
        if (i > TERRAIN_STARTPAD) velocity += rng.uniform(-1, 1) / SCALE;
        y += velocity;
      } else if (state === PIT && oneshot) {
        counter = rng.integers(3, 5);
        const poly = [[x, y], [x + TS, y], [x + TS, y - 4 * TS], [x, y - 4 * TS]];
        this._staticPoly(poly);
        const poly2 = poly.map(([px, py]) => [px + TS * counter, py]);
        this._staticPoly(poly2);
        this.obstacles.push({ kind: "pit", polys: [poly, poly2] });
        counter += 2;
        originalY = y;
      } else if (state === PIT && !oneshot) {
        y = originalY;
        if (counter > 1) y -= 4 * TS;
      } else if (state === STUMP && oneshot) {
        counter = rng.integers(1, 3);
        const poly = [[x, y], [x + counter * TS, y], [x + counter * TS, y + counter * TS], [x, y + counter * TS]];
        this._staticPoly(poly);
        this.obstacles.push({ kind: "stump", polys: [poly] });
      } else if (state === STAIRS && oneshot) {
        stairHeight = rng.random() > 0.5 ? +1 : -1;
        stairWidth = rng.integers(4, 5);
        stairSteps = rng.integers(3, 5);
        originalY = y;
        const polys = [];
        for (let s = 0; s < stairSteps; s++) {
          const poly = [
            [x + (s * stairWidth) * TS, y + (s * stairHeight) * TS],
            [x + ((1 + s) * stairWidth) * TS, y + (s * stairHeight) * TS],
            [x + ((1 + s) * stairWidth) * TS, y + (-1 + s * stairHeight) * TS],
            [x + (s * stairWidth) * TS, y + (-1 + s * stairHeight) * TS],
          ];
          this._staticPoly(poly);
          polys.push(poly);
        }
        this.obstacles.push({ kind: stairHeight > 0 ? "stairs_up" : "stairs_down", polys });
        counter = stairSteps * stairWidth;
      } else if (state === STAIRS && !oneshot) {
        const s = stairSteps * stairWidth - counter - stairHeight;
        const n = s / stairWidth;
        y = originalY + (n * stairHeight) * TS;
      }
      oneshot = false;
      this.terrainY.push(y);
      counter -= 1;
      if (counter === 0) {
        counter = rng.integers(Math.floor(TERRAIN_GRASS / 2), TERRAIN_GRASS);
        if (state === GRASS && this.hardcore) {
          state = rng.integers(1, N_STATES);
          oneshot = true;
        } else {
          state = GRASS;
          oneshot = true;
        }
      }
    }
    this.terrainPoly = [];
    for (let i = 0; i < TERRAIN_LENGTH - 1; i++) {
      const body = this.world.createBody({ type: "static" });
      body.createFixture({
        shape: planck.Edge(v2(this.terrainX[i], this.terrainY[i]), v2(this.terrainX[i + 1], this.terrainY[i + 1])),
        friction: f32(FRICTION), filterCategoryBits: 0x0001,
      });
      this.terrainPoly.push([[this.terrainX[i], this.terrainY[i]], [this.terrainX[i + 1], this.terrainY[i + 1]]]);
    }
  }

  _generateClouds() {
    const rng = this.rng, TS = TERRAIN_STEP;
    this.clouds = [];
    for (let k = 0; k < TERRAIN_LENGTH / 20; k++) {
      const x = rng.uniform(0, TERRAIN_LENGTH) * TS;
      const y = VIEWPORT_H / SCALE * 3 / 4;
      const poly = [];
      for (let a = 0; a < 5; a++) {
        const px = x + 15 * TS * Math.sin(3.14 * 2 * a / 5) + rng.uniform(0, 5 * TS);
        const py = y + 5 * TS * Math.cos(3.14 * 2 * a / 5) + rng.uniform(0, 5 * TS);
        poly.push([px, py]);
      }
      this.clouds.push(poly);
    }
  }

  _createRobot() {
    const initX = TERRAIN_STEP * TERRAIN_STARTPAD / 2;
    const initY = TERRAIN_HEIGHT + 2 * LEG_H;
    const robot = { filterCategoryBits: 0x0020, filterMaskBits: 0x001, restitution: 0 };
    this.hull = this.world.createBody({ type: "dynamic", position: v2(initX, initY) });
    this.hull.createFixture({ shape: planck.Polygon(HULL_POLY.map(([x, y]) => v2(x / SCALE, y / SCALE))),
                              density: 5.0, friction: f32(0.1), ...robot });
    this.hull.applyForceToCenter(v2(this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), true);
    this.legs = [];
    this.joints = [];
    for (const i of [-1, +1]) {
      const leg = this.world.createBody({ type: "dynamic", position: v2(initX, initY - LEG_H / 2 - LEG_DOWN), angle: f32(i * 0.05) });
      leg.createFixture({ shape: planck.Box(f32(LEG_W / 2), f32(LEG_H / 2)), density: 1.0, friction: f32(0.2), ...robot });
      this.joints.push(this.world.createJoint(planck.RevoluteJoint({
        localAnchorA: v2(0, LEG_DOWN), localAnchorB: v2(0, LEG_H / 2),
        enableMotor: true, enableLimit: true, maxMotorTorque: MOTORS_TORQUE, motorSpeed: i,
        lowerAngle: f32(-0.8), upperAngle: f32(1.1),
      }, this.hull, leg)));
      this.legs.push(leg);
      const lower = this.world.createBody({ type: "dynamic", position: v2(initX, initY - LEG_H * 3 / 2 - LEG_DOWN), angle: f32(i * 0.05) });
      lower.createFixture({ shape: planck.Box(f32(0.8 * LEG_W / 2), f32(LEG_H / 2)), density: 1.0, friction: f32(0.2), ...robot });
      this.joints.push(this.world.createJoint(planck.RevoluteJoint({
        localAnchorA: v2(0, -LEG_H / 2), localAnchorB: v2(0, LEG_H / 2),
        enableMotor: true, enableLimit: true, maxMotorTorque: MOTORS_TORQUE, motorSpeed: 1,
        lowerAngle: f32(-1.6), upperAngle: f32(-0.1),
      }, leg, lower)));
      lower.groundContact = false;
      this.legs.push(lower);
    }
  }

  _physicsStep(action) {
    // Set the motors the way Box2D 2.3 (Python's) does: a body that is asleep is woken (and its
    // "still for how long" clock restarts); a body that is awake is left alone. planck.js's own
    // setters skip unchanged values and restart the clock every time, which changes when a
    // motionless robot falls asleep.
    const speeds = [SPEED_HIP, SPEED_KNEE, SPEED_HIP, SPEED_KNEE];
    for (let j = 0; j < 4; j++) {
      const a = f32(action[j]), joint = this.joints[j];
      joint.m_motorSpeed = f32(speeds[j] * sign(a));
      joint.m_maxMotorTorque = f32(MOTORS_TORQUE * Math.min(1, Math.abs(a)));
      for (const body of [joint.getBodyA(), joint.getBodyB()]) if (!body.isAwake()) body.setAwake(true);
    }
    this.world.step(1.0 / FPS, 6 * 30, 2 * 30);

    const pos = this.hull.getPosition(), vel = this.hull.getLinearVelocity();
    const px = pos.x, py = pos.y;
    for (let i = 0; i < 10; i++) {
      const ray = this.lidar[i];
      ray.fraction = 1.0;
      ray.p1 = [px, py];
      ray.p2 = [px + Math.sin(1.5 * i / 10.0) * LIDAR_RANGE, py - Math.cos(1.5 * i / 10.0) * LIDAR_RANGE];
      this.world.rayCast(v2(...ray.p1), v2(...ray.p2), (fixture, point, normal, fraction) => {
        if ((fixture.getFilterCategoryBits() & 1) === 0) return -1;
        ray.p2 = [point.x, point.y];
        ray.fraction = fraction;
        return fraction;
      });
    }
    const J = this.joints;
    const state = [
      this.hull.getAngle(),
      2.0 * this.hull.getAngularVelocity() / FPS,
      0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
      0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
      J[0].getJointAngle(), J[0].getJointSpeed() / SPEED_HIP,
      J[1].getJointAngle() + 1.0, J[1].getJointSpeed() / SPEED_KNEE,
      this.legs[1].groundContact ? 1.0 : 0.0,
      J[2].getJointAngle(), J[2].getJointSpeed() / SPEED_HIP,
      J[3].getJointAngle() + 1.0, J[3].getJointSpeed() / SPEED_KNEE,
      this.legs[3].groundContact ? 1.0 : 0.0,
      ...this.lidar.map(l => l.fraction),
    ];
    this.scroll = px - VIEWPORT_W / SCALE / 5;

    // The official score. As in Python (NumPy 2), the energy cost is a float32 number, which
    // turns the step's reward into float32 too.
    const shaping = 130 * px / SCALE - 5.0 * Math.abs(state[0]);
    let reward = 0;
    if (this.prevShaping !== null) reward = shaping - this.prevShaping;
    this.prevShaping = shaping;
    for (let j = 0; j < 4; j++)
      reward = f32(f32(reward) - f32(f32(0.00035 * MOTORS_TORQUE) * Math.min(1, Math.abs(f32(action[j])))));
    let terminated = false;
    if (this.gameOver || px < 0) { reward = -100; terminated = true; }
    if (px > FINISH_X) terminated = true;
    return { obs: Float32Array.from(state), reward, terminated };
  }

  /* One step of the game (1/50 s) with the walker_env.py judge on top. */
  step(action) {
    const { obs, reward, terminated } = this._physicsStep(action);
    this.obs = obs;
    this.steps += 1;
    this.score += reward;
    const x = this.hull.getPosition().x;
    this.maxX = Math.max(this.maxX, x);
    if (terminated || this.steps >= this.timeLimit) {
      this.done = true;
      this.outcome = this.gameOver ? "fell" : x < 0 ? "backed_off" : x > FINISH_X ? "finished" : "timed_out";
      this.distance = progress(this.maxX);
      this.endX = x;
    }
    return { obs, reward, done: this.done };
  }

  /* Where each robot part is now, for drawing: [x, y, angle] for the hull and the four leg parts. */
  pose() {
    return [this.hull, ...this.legs].map(b => { const p = b.getPosition(); return [p.x, p.y, b.getAngle()]; });
  }
}
