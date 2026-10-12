"""PacBot Task 2A - Robust Turn & Run Solver (Zero-Collision)."""
from collections import deque
import json
import math

import paho.mqtt.client as mqtt

# ------------------------------------------------------------------ MQTT
MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "pacbot/sensors"
TOPIC_POSES = "pacbot/poses"
TOPIC_PELLETS = "pacbot/pellets"
TOPIC_RESULT = "pacbot/result"
TOPIC_WHEEL_VEL = "pacbot/wheel_vel"

# ------------------------------------------------------------------ maze
N = 15
PITCH = 0.22
WALL_HALF = 0.005
EXIT = (N - 1, N - 1, 3)

STEP = ((0, 1), (-1, 0), (0, -1), (1, 0))
DIR_ANGLES = (0.0, math.pi / 2, math.pi, -math.pi / 2)

LAYOUT = (
    "+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+",
    "|                           |       |               |       |",
    "+---+---+   +---+   +   +   +   +   +---+   +---+   +---+   +",
    "|           |           |   |   |       |       |           |",
    "+   +   +   +   +---+---+---+   +---+   +   +   +---+---+   +",
    "|   |           |               |   |   |   |   |   |       |",
    "+   +---+---+---+   +---+   +---+   +   +---+   +   +   +---+",
    "|   |               |               |       |   |           |",
    "+   +   +---+---+---+   +   +---+   +---+   +   +---+   +   +",
    "|   |       |           |       |   |       |       |       |",
    "+   +---+   +   +   +---+---+   +   +   +---+---+   +---+   +",
    "|           |   |       |       |   |       |       |   |   |",
    "+   +   +   +   +---+   +   +---+   +---+   +   +---+   +   +",
    "|   |       |       |   |   |   |       |       |       |   |",
    "+   +   +   +---+---+   +   +   +   +   +   +---+   +   +   +",
    "|   |   |           |       |       |           |   |       |",
    "+   +   +---+---+   +   +---+---+   +---+---+   +   +---+---+",
    "|   |   |           |           |       |       |           |",
    "+   +   +   +---+---+---+---+   +   +   +   +---+---+---+   +",
    "|   |   |               |       |   |   |           |       |",
    "+   +---+---+   +---+   +   +---+   +   +---+   +   +   +---+",
    "|   |       |   |           |           |       |   |   |   |",
    "+   +   +   +   +   +---+   +---+---+   +   +---+   +   +   +",
    "|   |   |           |   |               |   |               |",
    "+   +   +---+---+---+   +---+---+---+   +   +---+   +---+   +",
    "|   |       |                   |       |       |       |   |",
    "+   +   +   +   +   +   +---+   +   +---+---+   +---+   +   +",
    "|   |   |       |   |           |   |       |   |       |   |",
    "+   +   +---+---+   +   +   +   +   +---+   +   +---+---+   +",
    "|               |           |               |               |",
    "+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ E +",
)


def _wall_table():
    walls = {}
    for row in range(N):
        for col in range(N):
            r, c = 2 * row + 1, 4 * col
            walls[row, col, 0] = LAYOUT[r][c + 4] == "|"
            walls[row, col, 1] = LAYOUT[r - 1][c + 1:c + 4] == "---"
            walls[row, col, 2] = LAYOUT[r][c] == "|"
            walls[row, col, 3] = LAYOUT[r + 1][c + 1:c + 4] == "---"
    return walls


WALLS = _wall_table()


def inside(row, col):
    return 0 <= row < N and 0 <= col < N


def neighbours(cell):
    row, col = cell
    out = []
    for d in range(4):
        n = (row + STEP[d][0], col + STEP[d][1])
        if not WALLS[row, col, d] and inside(*n):
            out.append((d, n))
    return out


# ----------------------------------------------------------------- robot
WHEEL_R = 0.017
TRACK = 0.078
TICKS_PER_REV = 360
M_PER_TICK = 2 * math.pi * WHEEL_R / TICKS_PER_REV
WHEEL_MAX = 30.0
NO_HIT = 2.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wheels(v, omega):
    left = (v - omega * TRACK / 2) / WHEEL_R
    right = (v + omega * TRACK / 2) / WHEEL_R
    return clamp(left, -WHEEL_MAX, WHEEL_MAX), clamp(right, -WHEEL_MAX, WHEEL_MAX)


# ============================================================== planning

def bfs_cells(start, goal):
    if start == goal:
        return [start]
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for _, nbr in neighbours(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(path + [nbr])
    return []


def plan_route(start, heading, pellet):
    exit_cell = (14, 14)

    p1 = bfs_cells(start, pellet)
    p2 = bfs_cells(pellet, exit_cell)
    full_cells = p1 + p2[1:]

    directions = []
    for i in range(len(full_cells) - 1):
        dr = full_cells[i + 1][0] - full_cells[i][0]
        dc = full_cells[i + 1][1] - full_cells[i][1]
        directions.append(STEP.index((dr, dc)))

    actions = []
    curr_heading = heading
    idx = 0
    while idx < len(directions):
        target_dir = directions[idx]
        if curr_heading != target_dir:
            actions.append(("turn", target_dir))
            curr_heading = target_dir

        count = 0
        while idx < len(directions) and directions[idx] == curr_heading:
            count += 1
            idx += 1
        actions.append(("run", count))

    if curr_heading != 3:
        actions.append(("turn", 3))
    actions.append(("exit", 1.5))

    return actions


# ================================================================ solver

class Solver:
    def __init__(self):
        self.t = 0.0
        self.spawn = None
        self.pellet = None
        self.actions = None
        self.prev_ticks = None
        self.done = False

        self.theta = 0.0
        self.action_idx = 0
        self.dist_covered = 0.0
        self.settle = 0

    def log(self, text):
        print(f"[{self.t:7.2f}s] {text}", flush=True)

    def on_poses(self, msg):
        bot, d = msg.get("bot"), msg.get("dir")
        if isinstance(bot, list) and d is not None and self.actions is None:
            self.spawn = (d, tuple(bot))
            self.theta = DIR_ANGLES[d]
            self.log(f"spawn {self.spawn[1]} facing {self.spawn[0]}")

    def on_pellets(self, msg):
        pellet = msg.get("pellet")
        if pellet and self.actions is None:
            self.pellet = tuple(pellet)
            self.log(f"pellet {self.pellet}")

    def step(self, s):
        self.t += s["dt"]
        ticks = (s["ticks_l"], s["ticks_r"])
        if self.prev_ticks is None:
            self.prev_ticks = ticks

        d_left = (ticks[0] - self.prev_ticks[0]) * M_PER_TICK
        d_right = (ticks[1] - self.prev_ticks[1]) * M_PER_TICK
        self.prev_ticks = ticks

        d_center = (d_left + d_right) / 2.0
        turn_rate = s["gyro"][2]

        if self.done or self.spawn is None or self.pellet is None:
            return 0.0, 0.0

        if self.actions is None:
            self.actions = plan_route(self.spawn[1], self.spawn[0], self.pellet)
            self.log(f"plan ({len(self.actions)} actions): {self.actions}")

        # Integrate gyro
        self.theta = wrap(self.theta + turn_rate * s["dt"])

        # Stabilization pause between actions
        if self.settle > 0:
            self.settle -= 1
            return 0.0, 0.0

        if self.action_idx >= len(self.actions):
            return 0.0, 0.0

        act_type, act_val = self.actions[self.action_idx]

        # ---------------- 1. TURN IN PLACE ----------------
        if act_type == "turn":
            target_angle = DIR_ANGLES[act_val]
            err = wrap(target_angle - self.theta)

            if abs(err) < 0.02:
                self.theta = target_angle
                self.action_idx += 1
                self.dist_covered = 0.0
                self.settle = 4
                return 0.0, 0.0

            omega = clamp(2.2 * err, -1.0, 1.0)
            if abs(omega) < 0.18:
                omega = 0.18 if err > 0 else -0.18
            return wheels(0.0, omega)

        # ---------------- 2. RUN STRAIGHT ----------------
        elif act_type in ("run", "exit"):
            self.dist_covered += d_center
            target_dist = act_val * PITCH

            # Front-facing safety cutoff: triggers only when approaching the target distance
            sl = s.get("sl", NO_HIT)
            sr = s.get("sr", NO_HIT)
            front_wall_close = (
                act_type == "run"
                and self.dist_covered >= (target_dist - 0.05)
                and (sl < 0.078 or sr < 0.078)
            )

            if self.dist_covered >= target_dist or front_wall_close:
                self.action_idx += 1
                self.dist_covered = 0.0
                self.settle = 4
                return 0.0, 0.0

            v = 0.20

            # Cardinal lock
            target_angle = round(self.theta / (math.pi / 2)) * (math.pi / 2)
            heading_err = wrap(target_angle - self.theta)

            omega = clamp(5.0 * heading_err, -1.0, 1.0)
            return wheels(v, omega)

        return 0.0, 0.0


# ================================================================== main

def _client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def main():
    solver = Solver()
    client = _client()

    def send(cli, left, right):
        cli.publish(TOPIC_WHEEL_VEL, json.dumps({"left": float(left), "right": float(right)}))

    def on_message(cli, _userdata, msg):
        if msg.topic == TOPIC_SENSORS:
            send(cli, *solver.step(json.loads(msg.payload)))
        elif msg.topic == TOPIC_POSES:
            if msg.payload:
                solver.on_poses(json.loads(msg.payload))
        elif msg.topic == TOPIC_PELLETS:
            if msg.payload:
                solver.on_pellets(json.loads(msg.payload))
        elif msg.topic == TOPIC_RESULT:
            solver.log(f"result: {msg.payload.decode()}")
            solver.done = True
            send(cli, 0.0, 0.0)
            cli.disconnect()

    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    for topic in (TOPIC_POSES, TOPIC_PELLETS, TOPIC_RESULT, TOPIC_SENSORS):
        client.subscribe(topic)
    print("connected, waiting for the simulator", flush=True)
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
