

from __future__ import annotations
import math, random, os, argparse
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np



#   WORLD DATA STRUCTURES


@dataclass
class Obstacle:
    """Any dynamic or static object in the world (other vehicle, pedestrian…)."""
    x: float; y: float
    width:   float = 2.0
    height:  float = 4.5
    vx:      float = 0.0
    vy:      float = 0.0
    heading: float = 0.0
    category: str = "vehicle"   # vehicle | pedestrian | cone


@dataclass
class GPSFix:
    x: float; y: float
    heading: float; speed: float; timestamp: float



# 2  SENSOR PRIMITIVES


def _ray_aabb(ox, oy, dx, dy, cx, cy, hw, hh, max_d) -> Optional[float]:
    """Ray vs axis-aligned bounding box intersection. Returns distance or None."""
    inv_dx = 1.0 / (dx + 1e-12)
    inv_dy = 1.0 / (dy + 1e-12)
    t1x = ((cx - hw) - ox) * inv_dx;  t2x = ((cx + hw) - ox) * inv_dx
    t1y = ((cy - hh) - oy) * inv_dy;  t2y = ((cy + hh) - oy) * inv_dy
    te  = max(min(t1x, t2x), min(t1y, t2y))
    tx  = min(max(t1x, t2x), max(t1y, t2y))
    if te > tx or tx < 0 or te > max_d:
        return None
    return max(te, 0.0)


class LiDAR:
    """
    Rotating 2-D LiDAR: 360 evenly-spaced range beams over a full circle.
    Output: (num_beams,) float32 distances in [0, max_range] metres.
    """

    def __init__(self, num_beams: int = 360, max_range: float = 100.0,
                 noise_std: float = 0.05, dropout_prob: float = 0.01):
        self.num_beams = num_beams
        self.max_range = max_range
        self.noise_std = noise_std
        self.dropout   = dropout_prob
        self.angles    = np.linspace(0, 2 * math.pi, num_beams, endpoint=False)

    def scan(self, cx: float, cy: float, heading: float,
             obstacles: List[Obstacle],
             walls: List[Tuple[float, float, float, float]]) -> np.ndarray:
        r = np.full(self.num_beams, self.max_range, np.float32)
        for i, angle in enumerate(self.angles):
            wa = angle + heading
            dx, dy = math.cos(wa), math.sin(wa)
            for obs in obstacles:
                d = _ray_aabb(cx, cy, dx, dy, obs.x, obs.y,
                               obs.width / 2, obs.height / 2, self.max_range)
                if d is not None and d < r[i]:
                    r[i] = d
            for (wx, wy, ww, wh) in walls:
                d = _ray_aabb(cx, cy, dx, dy, wx + ww / 2, wy + wh / 2,
                               ww / 2, wh / 2, self.max_range)
                if d is not None and d < r[i]:
                    r[i] = d
        r += np.random.normal(0, self.noise_std, self.num_beams).astype(np.float32)
        r[np.random.random(self.num_beams) < self.dropout] = self.max_range
        return np.clip(r, 0, self.max_range)


class Radar:
    """
    Forward-looking FMCW radar.
    Output: (max_targets × 4,) flat float32 — [dist, angle, radial_vel, rcs] per target.
    """

    def __init__(self, max_range: float = 200.0, fov_deg: float = 60.0,
                 max_targets: int = 8, noise_dist: float = 0.3, noise_vel: float = 0.1):
        self.max_range   = max_range
        self.fov         = math.radians(fov_deg)
        self.max_targets = max_targets
        self.nd, self.nv = noise_dist, noise_vel

    def scan(self, cx: float, cy: float, heading: float,
             vx: float, vy: float,
             obstacles: List[Obstacle]) -> np.ndarray:
        targets = []
        for obs in obstacles:
            dx, dy = obs.x - cx, obs.y - cy
            dist = math.hypot(dx, dy)
            if dist > self.max_range or dist < 0.5:
                continue
            angle = (math.atan2(dy, dx) - heading + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) > self.fov / 2:
                continue
            ux, uy = dx / (dist + 1e-9), dy / (dist + 1e-9)
            rv = -((obs.vx - vx) * ux + (obs.vy - vy) * uy)
            targets.append([dist + np.random.normal(0, self.nd),
                             angle,
                             rv  + np.random.normal(0, self.nv),
                             obs.width * obs.height])
        targets.sort(key=lambda t: t[0])
        targets = targets[:self.max_targets]
        out = np.zeros(self.max_targets * 4, np.float32)
        for i, t in enumerate(targets):
            b = i * 4
            out[b]   = np.clip(t[0] / self.max_range, 0, 1)
            out[b+1] = t[1] / math.pi
            out[b+2] = np.clip(t[2] / 30.0, -1, 1)
            out[b+3] = np.clip(t[3] / 20.0,  0, 1)
        return out


class Ultrasonic:
    """
    8 ultrasonic proximity sensors at 45° intervals around the car body.
    Output: (8,) float32 distances in [0, max_range] metres.
    """

    SENSOR_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, max_range: float = 5.0, noise_std: float = 0.02,
                 car_hw: float = 1.0, car_hl: float = 2.25):
        self.max_range = max_range
        self.noise_std = noise_std
        self.car_hw    = car_hw
        self.car_hl    = car_hl

    def scan(self, cx: float, cy: float, heading: float,
             obstacles: List[Obstacle]) -> np.ndarray:
        r      = np.full(8, self.max_range, np.float32)
        offset = math.hypot(self.car_hl, self.car_hw) * 0.5
        for i, deg in enumerate(self.SENSOR_ANGLES):
            wa = heading + math.radians(deg)
            sx = cx + offset * math.cos(wa)
            sy = cy + offset * math.sin(wa)
            dx, dy = math.cos(wa), math.sin(wa)
            for obs in obstacles:
                d = _ray_aabb(sx, sy, dx, dy, obs.x, obs.y,
                               obs.width / 2, obs.height / 2, self.max_range)
                if d is not None and d < r[i]:
                    r[i] = d
        r += np.random.normal(0, self.noise_std, 8).astype(np.float32)
        return np.clip(r, 0, self.max_range)


class SemanticCamera:
    """
    Simulated bird's-eye semantic segmentation camera.

    Channels
    ────────
    ch0 : road presence   (0=off-road, 0.8=road)
    ch1 : obstacle mask   (0=empty,    0.9=obstacle)
    ch2 : goal heatmap    (Gaussian centered on next waypoint)

    Output: (grid_size, grid_size, 3) float32 image.
    """

    def __init__(self, grid_size: int = 64, range_m: float = 40.0,
                 noise_std: float = 0.02):
        self.g         = grid_size
        self.range_m   = range_m
        self.noise_std = noise_std

    def _w2p(self, wx, wy, cx, cy) -> Tuple[int, int]:
        px = int(((wx - cx) / self.range_m + 1.0) * 0.5 * self.g)
        py = int(((wy - cy) / self.range_m + 1.0) * 0.5 * self.g)
        return px, py

    def render(self, cx: float, cy: float, heading: float,
               obstacles: List[Obstacle],
               road_boxes: List[Tuple],
               waypoint: Tuple[float, float]) -> np.ndarray:
        g   = self.g
        img = np.zeros((g, g, 3), np.float32)
        img[:, :, 0] = 0.15   # default off-road

        for (rx, ry, rw, rh) in road_boxes:
            px0, py0 = self._w2p(rx,      ry,      cx, cy)
            px1, py1 = self._w2p(rx + rw, ry + rh, cx, cy)
            x0, x1 = sorted([px0, px1]); y0, y1 = sorted([py0, py1])
            x0,x1 = max(0,x0),min(g,x1); y0,y1 = max(0,y0),min(g,y1)
            img[y0:y1, x0:x1, 0] = 0.8

        for obs in obstacles:
            px, py = self._w2p(obs.x, obs.y, cx, cy)
            hw = max(1, int(obs.width  / self.range_m * g * 0.5))
            hh = max(1, int(obs.height / self.range_m * g * 0.5))
            x0,x1 = max(0,px-hw),min(g,px+hw); y0,y1 = max(0,py-hh),min(g,py+hh)
            img[y0:y1, x0:x1, 1] = 0.9

        gx, gy = self._w2p(waypoint[0], waypoint[1], cx, cy)
        yy, xx = np.mgrid[0:g, 0:g]
        sigma  = g * 0.08
        img[:, :, 2] = np.clip(
            np.exp(-((xx - gx)**2 + (yy - gy)**2) / (2 * sigma**2)).astype(np.float32),
            0, 1)
        img += np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)


class GPS:
    """
    GPS / localisation module with waypoint routing.
    Outputs a 6-dim navigation vector: [dx, dy, dist, cos_bearing, sin_bearing, speed_norm]
    """

    def __init__(self, pos_noise: float = 0.3,
                 heading_noise: float = 0.005,
                 speed_noise: float   = 0.1):
        self.pos_noise     = pos_noise
        self.heading_noise = heading_noise
        self.speed_noise   = speed_noise
        self.waypoints: List[Tuple[float, float]] = []
        self._idx = 0

    def load_route(self, waypoints):
        self.waypoints = list(waypoints)
        self._idx = 0

    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        return self.waypoints[self._idx] if self._idx < len(self.waypoints) else None

    def update_progress(self, cx: float, cy: float, threshold: float = 5.0) -> bool:
        """Advances to next waypoint when within threshold. Returns True at destination."""
        wp = self.current_waypoint()
        if wp is None:
            return True
        if math.hypot(cx - wp[0], cy - wp[1]) < threshold:
            self._idx += 1
        return self._idx >= len(self.waypoints)

    def fix(self, tx, ty, th, ts, t) -> GPSFix:
        return GPSFix(
            x        = tx + np.random.normal(0, self.pos_noise),
            y        = ty + np.random.normal(0, self.pos_noise),
            heading  = th + np.random.normal(0, self.heading_noise),
            speed    = max(0.0, ts + np.random.normal(0, self.speed_noise)),
            timestamp= t,
        )

    def nav_vector(self, fix: GPSFix) -> np.ndarray:
        wp = self.current_waypoint()
        if wp is None:
            return np.zeros(6, np.float32)
        dxw = wp[0] - fix.x; dyw = wp[1] - fix.y
        dist = math.hypot(dxw, dyw)
        ch, sh = math.cos(fix.heading), math.sin(fix.heading)
        dxc =  ch * dxw + sh * dyw
        dyc = -sh * dxw + ch * dyw
        bear = math.atan2(dyc, dxc)
        return np.array([
            np.clip(dxc / 200, -1, 1), np.clip(dyc / 200, -1, 1),
            np.clip(dist / 200,  0, 1),
            math.cos(bear), math.sin(bear),
            np.clip(fix.speed / 20, 0, 1),
        ], np.float32)


class SensorSuite:
    """
    Aggregates all sensors into a single observation.

    Scalar vector layout  (406 dims total)
    ───────────────────────────────────────
    [0   : 360]  LiDAR (norm 0→1)
    [360 : 392]  Radar (8 targets × 4 values)
    [392 : 400]  Ultrasonic (norm 0→1)
    [400 : 406]  GPS navigation vector

    Camera image (64 × 64 × 3) returned separately.
    """

    SCALAR_DIM = 406
    CAM_GRID   = 64
    CAM_DIM    = 64 * 64 * 3

    def __init__(self):
        self.lidar  = LiDAR(num_beams=360)
        self.radar  = Radar(max_targets=8)
        self.us     = Ultrasonic()
        self.camera = SemanticCamera(grid_size=self.CAM_GRID)
        self.gps    = GPS()

    def observe(self, cx, cy, heading, vx, vy, speed,
                obstacles, walls, road_boxes, t) -> Tuple[np.ndarray, np.ndarray]:
        fix = self.gps.fix(cx, cy, heading, speed, t)
        wp  = self.gps.current_waypoint() or (cx, cy)

        lidar_r = self.lidar.scan(cx, cy, heading, obstacles, walls) / self.lidar.max_range
        radar_r = self.radar.scan(cx, cy, heading, vx, vy, obstacles)
        us_r    = self.us.scan(cx, cy, heading, obstacles) / self.us.max_range
        nav_r   = self.gps.nav_vector(fix)
        cam_img = self.camera.render(cx, cy, heading, obstacles, road_boxes, wp)

        scalar = np.concatenate([lidar_r, radar_r, us_r, nav_r]).astype(np.float32)
        return scalar, cam_img



#   SIMULATION ENVIRONMENT


class RoadMap:
    """Simple grid road network: horizontal + vertical lane strips."""

    def __init__(self, size: float = 200.0, lane_width: float = 7.0):
        self.size = size
        s, lw = size, lane_width
        self.road_boxes: List[Tuple] = [
            (0, s*.15-lw/2, s, lw), (0, s*.35-lw/2, s, lw),
            (0, s*.65-lw/2, s, lw), (0, s*.85-lw/2, s, lw),
            (s*.15-lw/2, 0, lw, s), (s*.35-lw/2, 0, lw, s),
            (s*.65-lw/2, 0, lw, s), (s*.85-lw/2, 0, lw, s),
        ]
        self.walls: List[Tuple] = [
            (0, 0, s, 2), (0, s-2, s, 2), (0, 0, 2, s), (s-2, 0, 2, s),
        ]
        self.routes: List[List[Tuple[float, float]]] = [
            [(s*.15,s*.15),(s*.85,s*.15),(s*.85,s*.85),(s*.15,s*.85),(s*.15,s*.15)],
            [(s*.35,s*.35),(s*.65,s*.35),(s*.65,s*.65),(s*.35,s*.65),(s*.35,s*.35)],
            [(s*.15,s*.50),(s*.50,s*.50),(s*.85,s*.50),(s*.50,s*.50),(s*.50,s*.85)],
        ]

    def spawn_point(self, ridx: int = 0) -> Tuple[float, float, float]:
        w0, w1 = self.routes[ridx][0], self.routes[ridx][1]
        return w0[0], w0[1], math.atan2(w1[1]-w0[1], w1[0]-w0[0])

    def sample_traffic(self, n: int = 6) -> List[Obstacle]:
        obs = []
        for _ in range(n):
            rx,ry,rw,rh = random.choice(self.road_boxes)
            ox = rx + random.uniform(1, rw - 1)
            oy = ry + random.uniform(1, rh - 1)
            spd = random.uniform(0, 8)
            if rw > rh: vx,vy = spd*random.choice([-1,1]), 0.0
            else:       vx,vy = 0.0, spd*random.choice([-1,1])
            obs.append(Obstacle(ox, oy, vx=vx, vy=vy))
        return obs


class BicycleModel:
    """
    Kinematic bicycle model for ego-vehicle dynamics.
    Action: [steer ∈[-1,1], throttle ∈[0,1], brake ∈[0,1]]
    State:  (x, y, heading, speed)
    """

    def __init__(self, wheelbase=2.8, max_steer=0.6, max_acc=4.0,
                 max_brk=8.0, max_spd=20.0, drag=0.05):
        self.L    = wheelbase
        self.ms   = max_steer
        self.ma   = max_acc
        self.mb   = max_brk
        self.mspd = max_spd
        self.drag = drag

    def step(self, x, y, heading, speed, action, dt=0.1):
        steer = np.clip(action[0], -1, 1) * self.ms
        acc   = np.clip(action[1],  0, 1) * self.ma \
              - np.clip(action[2],  0, 1) * self.mb \
              - self.drag * speed
        ns    = np.clip(speed + acc * dt, 0.0, self.mspd)
        avg   = (speed + ns) / 2.0
        nh    = heading + avg / self.L * math.tan(steer) * dt
        nx    = x + avg * math.cos(nh) * dt
        ny    = y + avg * math.sin(nh) * dt
        return nx, ny, nh, float(ns)


class SelfDrivingEnv:
    """
    Gym-style self-driving simulation environment.

    Reward signals
    ──────────────
    +2.0 × v·cos(θ_err)       progress toward waypoint
    +50                        waypoint reached
    +200                       destination reached (terminal)
    −100                       collision (terminal)
    −0.5 / step                off-road
    −0.2 × Δv (over 50 km/h)  speed limit penalty
    −0.3 × Δaction²            jerk / comfort penalty
    −0.05 / step               idle (speed < 0.3 m/s)
    """

    MAX_STEPS   = 2000
    DT          = 0.1
    SPEED_LIMIT = 14.0   # m/s ≈ 50 km/h

    def __init__(self, map_size: float = 200.0, n_traffic: int = 6,
                 route_idx: Optional[int] = None):
        self.road      = RoadMap(map_size)
        self.n_traffic = n_traffic
        self.route_idx = route_idx
        self.sensors   = SensorSuite()
        self.dyn       = BicycleModel()
        self.x = self.y = self.heading = self.speed = 0.0
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.prev_action = np.zeros(3, np.float32)
        self.done = False
        self._prev_wp_idx = 0

    def _on_road(self, x, y) -> bool:
        for (rx,ry,rw,rh) in self.road.road_boxes:
            if rx<=x<=rx+rw and ry<=y<=ry+rh:
                return True
        return False

    def _collision(self) -> bool:
        m = self.road.size
        if self.x<2 or self.x>m-2 or self.y<2 or self.y>m-2:
            return True
        for o in self.obstacles:
            if abs(self.x-o.x)<(1+o.width/2) and abs(self.y-o.y)<(2.25+o.height/2):
                return True
        return False

    def _obs(self) -> Dict[str, np.ndarray]:
        sc, cam = self.sensors.observe(
            self.x, self.y, self.heading,
            self.speed * math.cos(self.heading),
            self.speed * math.sin(self.heading),
            self.speed, self.obstacles, self.road.walls,
            self.road.road_boxes, self.step_count * self.DT)
        return {"scalar": sc, "camera": cam}

    def reset(self) -> Dict[str, np.ndarray]:
        ridx = self.route_idx if self.route_idx is not None else \
               random.randrange(len(self.road.routes))
        self.x, self.y, self.heading = self.road.spawn_point(ridx)
        self.speed = 0.0
        self.obstacles = self.road.sample_traffic(self.n_traffic)
        self.sensors.gps.load_route(self.road.routes[ridx][1:])
        self.step_count = 0
        self.prev_action = np.zeros(3, np.float32)
        self.done = False
        self._prev_wp_idx = 0
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, np.float32)
        self.x, self.y, self.heading, self.speed = self.dyn.step(
            self.x, self.y, self.heading, self.speed, action, self.DT)
        for o in self.obstacles:
            o.x = (o.x + o.vx * self.DT) % self.road.size
            o.y = (o.y + o.vy * self.DT) % self.road.size

        reached    = self.sensors.gps.update_progress(self.x, self.y)
        wp         = self.sensors.gps.current_waypoint() or (self.x, self.y)
        cur_idx    = self.sensors.gps._idx

        # ── reward ──
        r = 0.0
        vec = np.array([wp[0]-self.x, wp[1]-self.y])
        d   = np.linalg.norm(vec) + 1e-8
        hv  = np.array([math.cos(self.heading), math.sin(self.heading)])
        r  += 2.0 * self.speed * float(np.dot(hv, vec / d))
        if cur_idx > self._prev_wp_idx:
            r += 50.0
        self._prev_wp_idx = cur_idx
        if reached:
            r += 200.0; self.done = True
        if self._collision():
            r -= 100.0; self.done = True
        if not self._on_road(self.x, self.y):
            r -= 0.5
        if self.speed > self.SPEED_LIMIT:
            r -= 0.2 * (self.speed - self.SPEED_LIMIT)
        r -= 0.3 * float(np.sum((action - self.prev_action)**2))
        if self.speed < 0.3:
            r -= 0.05
        self.prev_action = action.copy()
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS:
            self.done = True

        info = {
            "speed":       self.speed,
            "x": self.x,  "y": self.y,
            "heading":     self.heading,
            "waypoint_idx": cur_idx,
            "on_road":     self._on_road(self.x, self.y),
            "collision":   self._collision(),
        }
        return self._obs(), float(r), self.done, info



#   NEURAL NETWORK  (pure NumPy)

# ── Activations ───────────────────────────────────────────────────────────────

def relu(x):    return np.maximum(0.0, x)
def tanh(x):    return np.tanh(x)


# ── Parameter initialisers ────────────────────────────────────────────────────

def glorot_u(fi, fo):
    lim = math.sqrt(6.0 / (fi + fo))
    return np.random.uniform(-lim, lim, (fi, fo)).astype(np.float32)

def glorot_n(fi, fo):
    std = math.sqrt(2.0 / (fi + fo))
    return np.random.normal(0, std, (fi, fo)).astype(np.float32)


# ── Layer primitives ─────────────────────────────────────────────────────────

class Linear:
    def __init__(self, fi, fo):
        self.W = glorot_u(fi, fo)
        self.b = np.zeros(fo, np.float32)
    def forward(self, x):
        return x @ self.W + self.b
    def params(self):
        return [self.W, self.b]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps   = eps
        self.gamma = np.ones(dim, np.float32)
        self.beta  = np.zeros(dim, np.float32)
    def forward(self, x):
        mu  = x.mean(-1, keepdims=True)
        std = x.std(-1,  keepdims=True) + self.eps
        return self.gamma * (x - mu) / std + self.beta
    def params(self):
        return [self.gamma, self.beta]


class Conv1D:
    """(B, L, C) → (B, L', C_out), valid padding."""
    def __init__(self, k, ci, co, stride=1):
        lim = math.sqrt(6.0 / (k*ci + k*co))
        self.W = np.random.uniform(-lim, lim, (k, ci, co)).astype(np.float32)
        self.b = np.zeros(co, np.float32)
        self.k = k; self.s = stride
    def forward(self, x):
        B, L, C = x.shape
        ol = (L - self.k) // self.s + 1
        out = np.zeros((B, ol, self.W.shape[2]), np.float32)
        Wf  = self.W.reshape(-1, self.W.shape[2])
        for i in range(ol):
            patch = x[:, i*self.s:i*self.s+self.k, :]
            out[:, i, :] = patch.reshape(B, -1) @ Wf + self.b
        return out
    def params(self):
        return [self.W, self.b]


class Conv2D:
    """(B, H, W, C) → (B, H', W', C_out), valid padding."""
    def __init__(self, kh, kw, ci, co, stride=1):
        lim = math.sqrt(6.0 / (kh*kw*ci + kh*kw*co))
        self.W = np.random.uniform(-lim, lim, (kh, kw, ci, co)).astype(np.float32)
        self.b = np.zeros(co, np.float32)
        self.kh = kh; self.kw = kw; self.s = stride
    def forward(self, x):
        B, H, W, C = x.shape
        oh = (H - self.kh) // self.s + 1
        ow = (W - self.kw) // self.s + 1
        out = np.zeros((B, oh, ow, self.W.shape[3]), np.float32)
        Wf  = self.W.reshape(-1, self.W.shape[3])
        for i in range(oh):
            for j in range(ow):
                ri, rj = i*self.s, j*self.s
                patch = x[:, ri:ri+self.kh, rj:rj+self.kw, :]
                out[:, i, j, :] = patch.reshape(B, -1) @ Wf + self.b
        return out
    def params(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes, act=relu, out_act=None):
        self.layers  = [Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        self.norms   = [LayerNorm(sizes[i+1]) if i<len(sizes)-2 else None
                        for i in range(len(sizes)-1)]
        self.act     = act
        self.out_act = out_act
    def forward(self, x):
        for layer, norm in zip(self.layers[:-1], self.norms[:-1]):
            x = layer.forward(x)
            if norm: x = norm.forward(x)
            x = self.act(x)
        x = self.layers[-1].forward(x)
        if self.out_act: x = self.out_act(x)
        return x
    def params(self):
        p = []
        for l in self.layers: p.extend(l.params())
        for n in self.norms:
            if n: p.extend(n.params())
        return p


# ── Sensor encoders ───────────────────────────────────────────────────────────

class LiDAREncoder:
    """1-D CNN: (B, 360) → (B, 64)."""
    OUT = 64
    def __init__(self):
        self.c1 = Conv1D(9, 1, 16, stride=3)
        self.c2 = Conv1D(7, 16, 32, stride=3)
        self.c3 = Conv1D(5, 32, 64, stride=2)
        self.fc = Linear(64, self.OUT)
        self.ln = LayerNorm(self.OUT)
    def forward(self, x):
        B = x.shape[0]
        x = relu(self.c1.forward(x.reshape(B, 360, 1)))
        x = relu(self.c2.forward(x))
        x = relu(self.c3.forward(x))
        x = x.mean(axis=1)
        return relu(self.ln.forward(self.fc.forward(x)))
    def params(self):
        p = []
        for m in [self.c1,self.c2,self.c3,self.fc,self.ln]: p.extend(m.params())
        return p


class RadarEncoder:
    """MLP: (B, 32) → (B, 32)."""
    OUT = 32
    def __init__(self): self.mlp = MLP([32, 64, self.OUT])
    def forward(self, x): return self.mlp.forward(x)
    def params(self): return self.mlp.params()


class UltrasonicEncoder:
    """MLP: (B, 8) → (B, 16)."""
    OUT = 16
    def __init__(self): self.mlp = MLP([8, 32, self.OUT])
    def forward(self, x): return self.mlp.forward(x)
    def params(self): return self.mlp.params()


class CameraEncoder:
    """2-D CNN: (B, 64, 64, 3) → (B, 64)."""
    OUT = 64
    def __init__(self):
        self.c1 = Conv2D(5, 5, 3,  8,  stride=2)
        self.c2 = Conv2D(4, 4, 8,  16, stride=2)
        self.c3 = Conv2D(3, 3, 16, 32, stride=2)
        self.fc = Linear(32, self.OUT)
    def forward(self, x):
        x = relu(self.c1.forward(x))
        x = relu(self.c2.forward(x))
        x = relu(self.c3.forward(x))
        x = x.max(axis=(1, 2))      # global max pool
        return relu(self.fc.forward(x))
    def params(self):
        p = []
        for m in [self.c1,self.c2,self.c3,self.fc]: p.extend(m.params())
        return p


class GPSEncoder:
    """MLP: (B, 6) → (B, 16)."""
    OUT = 16
    def __init__(self): self.mlp = MLP([6, 32, self.OUT])
    def forward(self, x): return self.mlp.forward(x)
    def params(self): return self.mlp.params()


# ── Fusion + Actor-Critic ─────────────────────────────────────────────────────

class SensorFusionNetwork:
    """Multi-modal fusion: (scalar(B,406), cam(B,64,64,3)) → (B, 128)."""
    FUSED = 128
    def __init__(self):
        self.lidar_enc = LiDAREncoder()         # 360 → 64
        self.radar_enc = RadarEncoder()         #  32 → 32
        self.us_enc    = UltrasonicEncoder()    #   8 → 16
        self.cam_enc   = CameraEncoder()        # 64×64×3 → 64
        self.gps_enc   = GPSEncoder()           #   6 → 16
        self.fusion    = MLP([192, 256, self.FUSED])   # 64+32+16+64+16 = 192
    def forward(self, sc, cam):
        f = np.concatenate([
            self.lidar_enc.forward(sc[:, :360]),
            self.radar_enc.forward(sc[:, 360:392]),
            self.us_enc.forward(sc[:, 392:400]),
            self.cam_enc.forward(cam),
            self.gps_enc.forward(sc[:, 400:406]),
        ], axis=-1)
        return self.fusion.forward(f)
    def params(self):
        p = []
        for e in [self.lidar_enc,self.radar_enc,self.us_enc,
                  self.cam_enc,self.gps_enc,self.fusion]:
            p.extend(e.params())
        return p


LOG_STD_MIN, LOG_STD_MAX = -4.0, 0.5

class ActorHead:
    """Gaussian policy: feat → (mean(3), log_std(3))."""
    def __init__(self, fd=128):
        self.mean_mlp    = MLP([fd, 64, 3], out_act=tanh)
        self.logstd_mlp  = MLP([fd, 32, 3])
    def forward(self, feat):
        mean    = self.mean_mlp.forward(feat)
        log_std = np.clip(self.logstd_mlp.forward(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    def sample(self, feat):
        mean, ls = self.forward(feat)
        std = np.exp(ls)
        raw = mean + np.random.normal(0,1,mean.shape).astype(np.float32) * std
        action = np.stack([np.clip(raw[:,0],-1,1),
                           np.clip(raw[:,1], 0,1),
                           np.clip(raw[:,2], 0,1)], axis=-1)
        lp = self._lp(raw, mean, ls)
        return action, lp
    def log_prob(self, feat, action):
        mean, ls = self.forward(feat)
        return self._lp(action, mean, ls)
    @staticmethod
    def _lp(x, mean, log_std):
        std = np.exp(log_std)
        lp  = -0.5 * ((x - mean) / (std + 1e-8))**2 \
              - log_std - 0.5 * math.log(2 * math.pi)
        return lp.sum(axis=-1)
    def params(self):
        return self.mean_mlp.params() + self.logstd_mlp.params()


class CriticHead:
    """Value network: feat → scalar."""
    def __init__(self, fd=128): self.mlp = MLP([fd, 64, 1])
    def forward(self, feat): return self.mlp.forward(feat).squeeze(-1)
    def params(self): return self.mlp.params()


class PPOModel:
    """
    Full model: SensorFusion + Actor + Critic.
    ~137 K parameters, pure NumPy.
    """
    def __init__(self):
        self.fusion = SensorFusionNetwork()
        self.actor  = ActorHead(SensorFusionNetwork.FUSED)
        self.critic = CriticHead(SensorFusionNetwork.FUSED)

    def _prep(self, sc, cam):
        sc  = np.asarray(sc, np.float32);  sc  = sc[np.newaxis]  if sc.ndim==1  else sc
        cam = np.asarray(cam, np.float32); cam = cam[np.newaxis] if cam.ndim==3 else cam
        return sc, cam

    def act(self, sc, cam):
        s, c = self._prep(sc, cam)
        feat = self.fusion.forward(s, c)
        a, lp = self.actor.sample(feat)
        return a[0], float(lp[0])

    def value(self, sc, cam):
        s, c = self._prep(sc, cam)
        feat = self.fusion.forward(s, c)
        return float(self.critic.forward(feat)[0])

    def log_prob(self, sc, cam, action):
        s, c = self._prep(sc, cam)
        a    = np.asarray(action, np.float32)
        a    = a[np.newaxis] if a.ndim==1 else a
        feat = self.fusion.forward(s, c)
        return self.actor.log_prob(feat, a)

    def all_params(self):
        return self.fusion.params() + self.actor.params() + self.critic.params()

    def param_count(self):
        return sum(p.size for p in self.all_params())

    def save(self, path: str):
        np.save(path, np.array([p.copy() for p in self.all_params()], object),
                allow_pickle=True)
        print(f"[Model] Saved → {path}")

    def load(self, path: str):
        saved = np.load(path, allow_pickle=True)
        for p, s in zip(self.all_params(), saved): p[:] = s
        print(f"[Model] Loaded ← {path}")



# §5  ADAM OPTIMISER

class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, clip=0.5):
        self.params = params
        self.lr     = lr
        self.b1,self.b2,self.eps,self.clip = b1,b2,eps,clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        norm = math.sqrt(sum((g**2).sum() for g in grads) + 1e-9)
        if norm > self.clip:
            grads = [g * self.clip / norm for g in grads]
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        for i,(p,g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g**2
            p -= self.lr * (self.m[i]/bc1) / (np.sqrt(self.v[i]/bc2) + self.eps)



#   ROLLOUT BUFFER + GAE


class RolloutBuffer:
    def __init__(self, cap, scalar_dim, cam_shape):
        self.cap     = cap
        self.scalars = np.zeros((cap, scalar_dim),  np.float32)
        self.cameras = np.zeros((cap, *cam_shape),  np.float32)
        self.actions = np.zeros((cap, 3),            np.float32)
        self.lps     = np.zeros(cap,                 np.float32)
        self.rewards = np.zeros(cap,                 np.float32)
        self.values  = np.zeros(cap,                 np.float32)
        self.dones   = np.zeros(cap,                 np.float32)
        self.ptr = 0; self.full = False

    def add(self, sc, cam, act, lp, rew, val, done):
        i = self.ptr
        self.scalars[i]=sc; self.cameras[i]=cam; self.actions[i]=act
        self.lps[i]=lp; self.rewards[i]=rew; self.values[i]=val; self.dones[i]=done
        self.ptr = (i+1) % self.cap
        if self.ptr == 0: self.full = True

    def size(self): return self.cap if self.full else self.ptr

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        n = self.size(); adv = np.zeros(n, np.float32); la = 0.0
        for t in reversed(range(n)):
            nv = last_val*(1-self.dones[t]) if t==n-1 else self.values[t+1]*(1-self.dones[t])
            delta = self.rewards[t] + gamma*nv - self.values[t]
            la = delta + gamma*lam*(1-self.dones[t])*la
            adv[t] = la
        return adv, adv + self.values[:n]

    def prepare(self, last_val, gamma, lam):
        self._adv, self._ret = self.compute_gae(last_val, gamma, lam)
        self._adv = (self._adv - self._adv.mean()) / (self._adv.std() + 1e-8)

    def batches(self, bs):
        n = self.size(); idx = np.random.permutation(n)
        for s in range(0, n-bs+1, bs):
            b = idx[s:s+bs]
            yield (self.scalars[b],self.cameras[b],self.actions[b],
                   self.lps[b],self._adv[b],self._ret[b])

    def reset(self): self.ptr = 0; self.full = False



# PPO AGENT


class PPOAgent:
    """
    PPO-Clip agent.

    Hyperparameters (defaults)
    ──────────────────────────
    gamma      = 0.99   discount
    gae_lambda = 0.95   GAE smoothing
    clip_eps   = 0.20   PPO clip range
    vf_coef    = 0.50   value loss coefficient
    ent_coef   = 0.01   entropy bonus coefficient
    lr         = 3e-4   Adam learning rate
    n_epochs   = 4      update epochs per rollout
    batch_size = 64     mini-batch size
    rollout_len= 512    steps per rollout
    """

    def __init__(self, scalar_dim=406, cam_shape=(64,64,3),
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 vf_coef=0.5, ent_coef=0.01, lr=3e-4,
                 n_epochs=4, batch_size=64, rollout_len=512):
        self.gamma      = gamma
        self.lam        = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        self.model  = PPOModel()
        self.optim  = Adam(self.model.all_params(), lr=lr)
        self.buffer = RolloutBuffer(rollout_len, scalar_dim, cam_shape)

        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.pol_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)
        self.update_count = 0

    def select_action(self, obs):
        a, lp = self.model.act(obs["scalar"], obs["camera"])
        v     = self.model.value(obs["scalar"], obs["camera"])
        return a, lp, v

    def store(self, obs, action, lp, reward, value, done):
        self.buffer.add(obs["scalar"], obs["camera"],
                        action, lp, reward, value, float(done))

    def update(self, last_obs):
        last_val = self.model.value(last_obs["scalar"], last_obs["camera"])
        self.buffer.prepare(last_val, self.gamma, self.lam)

        total_pl = total_vl = n = 0.0
        for _ in range(self.n_epochs):
            for sc, cam, act, old_lp, adv, ret in self.buffer.batches(self.batch_size):
                feat   = self.model.fusion.forward(sc, cam)
                values = self.model.critic.forward(feat)
                new_lp = self.model.actor.log_prob(feat, act)

                # ── critic gradient (output layer only) ──
                val_err  = values - ret
                feat_h   = relu(self.model.critic.mlp.layers[-2].forward(feat))
                dL_dW_v  = (feat_h.T @ val_err[:,None]) * (2*self.vf_coef/len(val_err))
                dL_db_v  = np.full_like(self.model.critic.mlp.layers[-1].b,
                                        val_err.mean()*2*self.vf_coef)

                # ── actor gradient (output layer only) ──
                ratio   = np.exp(np.clip(new_lp - old_lp, -10, 10))
                clipped = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
                r_eff   = np.where(ratio*adv < clipped*adv, ratio, clipped)
                mean, ls= self.model.actor.forward(feat)
                std     = np.exp(ls)
                dL_dm   = -(r_eff[:,None] * (act-mean) / (std**2+1e-8)) / len(adv)
                feat_a  = tanh(self.model.actor.mean_mlp.layers[-2].forward(feat))
                dL_dW_a = feat_a.T @ dL_dm
                dL_db_a = dL_dm.mean(axis=0)

                # Build gradient list
                all_p = self.model.all_params()
                grads = [np.zeros_like(p) for p in all_p]
                cW = self.model.critic.mlp.layers[-1].W
                cb = self.model.critic.mlp.layers[-1].b
                aW = self.model.actor.mean_mlp.layers[-1].W
                ab = self.model.actor.mean_mlp.layers[-1].b
                for i, p in enumerate(all_p):
                    if p is cW: grads[i] = dL_dW_v
                    elif p is cb: grads[i] = dL_db_v
                    elif p is aW: grads[i] = dL_dW_a
                    elif p is ab: grads[i] = dL_db_a

                self.optim.step(grads)

                pl = float(-np.mean(np.minimum(ratio*adv, clipped*adv)))
                vl = float(self.vf_coef * np.mean(val_err**2))
                total_pl += pl; total_vl += vl; n += 1

        self.buffer.reset()
        self.update_count += 1
        if n:
            self.pol_losses.append(total_pl/n)
            self.val_losses.append(total_vl/n)

    def log_episode(self, rew, length):
        self.ep_rewards.append(rew)
        self.ep_lengths.append(length)

    def stats(self) -> Dict:
        return {
            "mean_reward": float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0,
            "mean_length": float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0,
            "policy_loss": float(np.mean(self.pol_losses)) if self.pol_losses else 0.0,
            "value_loss":  float(np.mean(self.val_losses)) if self.val_losses else 0.0,
            "updates":     self.update_count,
        }



#  TRAINING & EVALUATION


def train(cfg: Dict):
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    ckpt_dir = cfg["checkpoint_dir"]

    env = SelfDrivingEnv(
        map_size  = cfg.get("map_size",   200.0),
        n_traffic = cfg.get("n_traffic",  4),
    )
    agent = PPOAgent(
        scalar_dim  = 406,
        cam_shape   = (64, 64, 3),
        gamma       = cfg.get("gamma",       0.99),
        gae_lambda  = cfg.get("gae_lambda",  0.95),
        clip_eps    = cfg.get("clip_eps",    0.20),
        vf_coef     = cfg.get("vf_coef",     0.50),
        ent_coef    = cfg.get("ent_coef",    0.01),
        lr          = cfg.get("lr",          3e-4),
        n_epochs    = cfg.get("n_epochs",    4),
        batch_size  = cfg.get("batch_size",  64),
        rollout_len = cfg.get("rollout_len", 512),
    )
    if cfg.get("load"):
        agent.model.load(cfg["load"])

    total_ep  = cfg.get("total_episodes", 500)
    log_every = cfg.get("log_interval",   10)
    save_every= cfg.get("save_interval",  50)

    print("=" * 65)
    print("  Self-Driving RL  –  PPO Agent")
    print(f"  Parameters   : {agent.model.param_count():,}")
    print(f"  Rollout len  : {cfg.get('rollout_len', 512)}")
    print(f"  Episodes     : {total_ep}")
    print("=" * 65)

    global_step = steps_since_upd = 0
    best_reward = -float("inf")

    for ep in range(1, total_ep + 1):
        obs = env.reset()
        ep_r = 0.0; done = False
        while not done:
            a, lp, v = agent.select_action(obs)
            obs2, r, done, info = env.step(a)
            agent.store(obs, a, lp, r, v, done)
            ep_r += r; global_step += 1; steps_since_upd += 1
            obs = obs2
            if steps_since_upd >= cfg.get("rollout_len", 512):
                agent.update(obs); steps_since_upd = 0
        agent.log_episode(ep_r, env.step_count)

        if ep % log_every == 0:
            s = agent.stats()
            print(f"Ep {ep:5d}/{total_ep} | "
                  f"R {s['mean_reward']:+8.1f} | "
                  f"Len {s['mean_length']:6.0f} | "
                  f"π {s['policy_loss']:.4f} | "
                  f"V {s['value_loss']:.4f} | "
                  f"Steps {global_step:,}")

        if ep % save_every == 0:
            agent.model.save(os.path.join(ckpt_dir, f"ep{ep}.npy"))

        if ep_r > best_reward:
            best_reward = ep_r
            agent.model.save(os.path.join(ckpt_dir, "best.npy"))

    final = os.path.join(ckpt_dir, "final.npy")
    agent.model.save(final)
    print(f"\n✓ Training complete. Best reward: {best_reward:.1f}")
    return agent


def evaluate(path: str, n_episodes: int = 10):
    env   = SelfDrivingEnv(n_traffic=4)
    agent = PPOAgent()
    agent.model.load(path)
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset(); total = 0.0; done = False
        while not done:
            a, _, _ = agent.select_action(obs)
            obs, r, done, info = env.step(a)
            total += r
        rewards.append(total)
        print(f"  Eval ep {ep+1:3d}: reward={total:+8.1f}  "
              f"speed={info['speed']:.1f}m/s  "
              f"on_road={'Y' if info['on_road'] else 'N'}")
    print(f"\nMean reward: {np.mean(rewards):.2f}  ±{np.std(rewards):.2f}")


def demo(n_steps: int = 300):
    """Single episode with step-by-step printout."""
    env   = SelfDrivingEnv(n_traffic=3)
    agent = PPOAgent()
    obs   = env.reset()
    total = 0.0
    print("Running demo episode...")
    print(f"{'Step':>5} {'Speed':>7} {'Reward':>8} {'WP':>4} {'OnRoad':>7} {'Collision':>10}")
    print("-" * 50)
    for s in range(n_steps):
        a, _, _ = agent.select_action(obs)
        obs, r, done, info = env.step(a)
        total += r
        if s % 20 == 0:
            print(f"{s:5d} {info['speed']:7.2f} {r:+8.3f} "
                  f"{info['waypoint_idx']:4d} "
                  f"{'YES':>7}" if info['on_road'] else
                  f"{s:5d} {info['speed']:7.2f} {r:+8.3f} "
                  f"{info['waypoint_idx']:4d} {'NO':>7}"
                  f" {'⚠ COLLISION' if info['collision'] else '':>10}")
        if done:
            print(f"\nEpisode ended at step {s}. Total reward: {total:.2f}")
            break
    else:
        print(f"\n{n_steps} steps complete. Total reward: {total:.2f}")



#   ENTRY POINT


def _parse():
    p = argparse.ArgumentParser(
        description="Self-Driving RL  –  PPO with multi-modal sensor fusion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--episodes",   type=int,   default=500,  help="Training episodes")
    p.add_argument("--lr",         type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--rollout",    type=int,   default=512,  help="Rollout length")
    p.add_argument("--batch",      type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--epochs",     type=int,   default=4,    help="PPO update epochs")
    p.add_argument("--traffic",    type=int,   default=4,    help="Number of traffic cars")
    p.add_argument("--log",        type=int,   default=10,   help="Log every N episodes")
    p.add_argument("--save",       type=int,   default=50,   help="Save every N episodes")
    p.add_argument("--ckpt",       type=str,   default="checkpoints",
                   help="Checkpoint directory")
    p.add_argument("--load",       type=str,   default=None,
                   help="Resume training from checkpoint")
    p.add_argument("--eval",       type=str,   default=None,
                   help="Evaluate a saved checkpoint (no training)")
    p.add_argument("--demo",       action="store_true",
                   help="Run one demo episode with verbose output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.demo:
        demo()
    elif args.eval:
        evaluate(args.eval)
    else:
        train({
            "total_episodes": args.episodes,
            "lr":             args.lr,
            "rollout_len":    args.rollout,
            "batch_size":     args.batch,
            "n_epochs":       args.epochs,
            "n_traffic":      args.traffic,
            "log_interval":   args.log,
            "save_interval":  args.save,
            "checkpoint_dir": args.ckpt,
            "load":           args.load,
        })
