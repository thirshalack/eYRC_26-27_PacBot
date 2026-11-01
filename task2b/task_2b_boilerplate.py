"""Boilerplate for PB Task 2B.

Autonomous collision-free and ghost-safe controller for PacBot Task 2B.
Integrates exact closed-loop laser touchdown, pure cardinal heading hold,
full-cycle ghost loop identification, and synchronized safe entry/exit gating.
"""
from collections import deque
import heapq
import itertools
import json
import math
import sys

import paho.mqtt.client as mqtt

import maze_2b
from maze_2b import Maze

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "pacbot/sensors"
TOPIC_PELLETS = "pacbot/pellets"
TOPIC_GHOST = "pacbot/ghost"
TOPIC_RESULT = "pacbot/result"
TOPIC_WHEEL_VEL = "pacbot/wheel_vel"

WHEEL_RADIUS = 0.017        # m
HALF_TRACK = 0.039          # m
AXLE_OFFSET = 0.033         # m behind chassis center
WHEEL_LIMIT = 30.0          # rad/s
TICKS_PER_REV = 360
TICK_RAD = 2.0 * math.pi / TICKS_PER_REV
CELL_PITCH = 0.22           # m

# Driving speeds - calibrated to eliminate corridor rubbing
CRUISE_SPEED = 18.0         # rad/s (~0.306 m/s)
APPROACH_SPEED = 4.0        # rad/s
START_SPEED = 8.0           # rad/s
TURN_TOLERANCE = 0.010      # rad (~0.57 deg)

CARDINAL_HEADINGS = {
    0: 0.0,
    1: math.pi / 2.0,
    2: math.pi,
    3: -math.pi / 2.0
}


def _mqtt_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def get_target_wall_dist(maze, cell, heading):
    curr = cell
    k = 0
    dr = [0, -1, 0, 1][heading]
    dc = [1, 0, -1, 0][heading]
    while maze.can_move(curr, heading):
        curr = (curr[0] + dr, curr[1] + dc)
        k += 1
    return k * CELL_PITCH + 0.077


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def extract_patrol_loop(history):
    cells = []
    for h in history:
        cell = h[0]
        if not cells or cells[-1] != cell:
            cells.append(cell)
    if len(cells) < 6:
        return None
    for i in range(len(cells)):
        c = cells[i]
        for j in range(i + 5, len(cells)):
            if cells[j] == c:
                cand = cells[i:j]
                if len(cand) == len(set(cand)) and len(cand) >= 5:
                    return cand
    return None


def dijkstra_safe(maze, start_cell, target_cell, ghost_cells=None, forbidden_edges=None):
    if start_cell == target_cell:
        return [start_cell]
    pq = [(0, start_cell, [start_cell])]
    visited = {}
    while pq:
        cost, curr, path = heapq.heappop(pq)
        if curr == target_cell:
            return path
        if curr in visited and visited[curr] <= cost:
            continue
        visited[curr] = cost
        for heading, nxt in maze.neighbours(curr):
            if forbidden_edges and (curr, nxt) in forbidden_edges:
                continue
            step_cost = 1
            if ghost_cells and nxt in ghost_cells and nxt != target_cell:
                step_cost = 50
            heapq.heappush(pq, (cost + step_cost, nxt, path + [nxt]))
    return None


def count_turns(cells):
    h = 0
    turns = 0
    for i in range(len(cells) - 1):
        dr = cells[i + 1][0] - cells[i][0]
        dc = cells[i + 1][1] - cells[i][1]
        if dr == 0 and dc == 1: req_h = 0
        elif dr == -1 and dc == 0: req_h = 1
        elif dr == 0 and dc == -1: req_h = 2
        elif dr == 1 and dc == 0: req_h = 3
        else: req_h = h
        if req_h != h:
            diff = abs(req_h - h)
            if diff == 3: diff = 1
            turns += diff
            h = req_h
    return turns


class Driver:
    def __init__(self, maze):
        self.maze = maze
        self.pellets = []
        self.initial_pellets = []
        self.have_pellets = False

        self._ghost = None
        self.ghost_history = deque(maxlen=3000)
        self.ghost_patrol_cells = set()
        self.directed_ghost_edges = set()
        self.forbidden_edges = set()
        self.guarded_pellet = None

        self.current_heading = 0
        self.theta = 0.0
        self.axle_x = -0.15 - AXLE_OFFSET
        self.axle_y = 0.00
        self.prev_ticks = None

        self.actions = []
        self.action_idx = 0
        self.planned = False

        self.standoff_action_idx = None
        self.entry_cell = None
        self.standoff_cleared = False
        self.standoff_ghost_was_close = False
        self.last_dist_to_entry = float("inf")

        self.state = "INIT"
        self.init_frames = 0
        self.settle_frames = 0
        self.brake_frames = 0
        self.seg_start_dist = 0.0
        self.total_traveled = 0.0

        self.result = None
        self.sim_time = 0.0

    @property
    def ghost(self):
        return self._ghost

    @ghost.setter
    def ghost(self, msg):
        self.on_ghost(msg)

    @property
    def chassis_x(self):
        return self.axle_x + AXLE_OFFSET * math.cos(self.theta)

    @property
    def chassis_y(self):
        return self.axle_y + AXLE_OFFSET * math.sin(self.theta)

    def on_pellets(self, msg):
        payload_pellets = [tuple(c) for c in msg.get("pellets", [])]
        if len(payload_pellets) == 3:
            self.pellets = list(payload_pellets)
            self.initial_pellets = list(payload_pellets)
            self.have_pellets = True
            print(f"[INIT] Pellets received: {self.initial_pellets}", flush=True)
        elif self.have_pellets:
            self.pellets = list(payload_pellets)

    def on_ghost(self, msg):
        self._ghost = msg
        if msg is not None:
            cell = (msg["row"], msg["col"])
            self.ghost_history.append((cell, msg["x"], msg["y"]))
            self.ghost_patrol_cells.add(cell)

    def plan_route(self):
        loop = extract_patrol_loop(self.ghost_history)
        if loop:
            self.ghost_patrol_cells = set(loop)
            self.directed_ghost_edges = set((loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop)))
            self.forbidden_edges = set((c2, c1) for (c1, c2) in self.directed_ghost_edges)
            for c1 in self.ghost_patrol_cells:
                for h, c2 in self.maze.neighbours(c1):
                    if c2 in self.ghost_patrol_cells and (c1, c2) not in self.directed_ghost_edges:
                        self.forbidden_edges.add((c1, c2))
        else:
            self.ghost_patrol_cells = set(h[0] for h in self.ghost_history)
            self.directed_ghost_edges = set()
            self.forbidden_edges = set()

        if self.ghost_patrol_cells and self.initial_pellets:
            best_p = None
            min_dist = float("inf")
            for p in self.initial_pellets:
                d = min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in self.ghost_patrol_cells)
                if d < min_dist:
                    min_dist = d
                    best_p = p
            self.guarded_pellet = best_p

        spawn_cell = self.maze.spawn_cell
        exit_cell = self.maze.exit_cell

        free_pellets = [p for p in self.initial_pellets if p != self.guarded_pellet]
        if len(free_pellets) < 2 and self.initial_pellets:
            free_pellets = self.initial_pellets[:2]
            p_guarded = self.initial_pellets[2]
        else:
            p_guarded = self.guarded_pellet

        orders = [
            (free_pellets[0], free_pellets[1], p_guarded),
            (free_pellets[1], free_pellets[0], p_guarded)
        ]

        best_path = None
        best_cost = float("inf")
        best_order = None

        for perm in orders:
            p1, p2, p3 = perm
            path1 = dijkstra_safe(self.maze, spawn_cell, p1, self.ghost_patrol_cells, self.forbidden_edges)
            path2 = dijkstra_safe(self.maze, p1, p2, self.ghost_patrol_cells, self.forbidden_edges)
            path3 = dijkstra_safe(self.maze, p2, p3, self.ghost_patrol_cells, self.forbidden_edges)
            path4 = dijkstra_safe(self.maze, p3, exit_cell, self.ghost_patrol_cells, self.forbidden_edges)

            full = path1 + path2[1:] + path3[1:] + path4[1:]
            turns = count_turns(full)
            cost = len(full) + 0.6 * turns

            if cost < best_cost:
                best_cost = cost
                best_order = perm
                best_path = full

        self.actions = [("CHUTE", 0.00)]
        curr_h = 0

        i = 0
        while i < len(best_path) - 1:
            c1 = best_path[i]
            c2 = best_path[i + 1]
            dr = c2[0] - c1[0]
            dc = c2[1] - c1[1]

            if dr == 0 and dc == 1: req_h = 0
            elif dr == -1 and dc == 0: req_h = 1
            elif dr == 0 and dc == -1: req_h = 2
            elif dr == 1 and dc == 0: req_h = 3
            else: req_h = curr_h

            if req_h != curr_h:
                self.actions.append(("TURN", req_h))
                curr_h = req_h

            j = i
            while j < len(best_path) - 1:
                if j > i and best_path[j] not in self.ghost_patrol_cells and best_path[j + 1] in self.ghost_patrol_cells:
                    break
                if c1 not in self.ghost_patrol_cells and c2 in self.ghost_patrol_cells and j > i:
                    break
                nj_dr = best_path[j + 1][0] - best_path[j][0]
                nj_dc = best_path[j + 1][1] - best_path[j][1]
                if (nj_dr, nj_dc) == (dr, dc):
                    j += 1
                else:
                    break

            end_cell = best_path[j]
            wall_ahead = not self.maze.can_move(end_cell, req_h)
            is_standoff = (c1 not in self.ghost_patrol_cells and c2 in self.ghost_patrol_cells)
            self.actions.append(("DRIVE_SEG", req_h, end_cell, wall_ahead, is_standoff))
            i = j

        self.actions.append(("EXIT", 3.25))

        self.standoff_action_idx = None
        self.entry_cell = None
        p2_found = False
        p2 = best_order[1]
        for a_idx, act in enumerate(self.actions):
            if act[0] == "DRIVE_SEG":
                req_h, end_cell, wall_ahead, is_standoff = act[1], act[2], act[3], act[4]
                if end_cell == p2:
                    p2_found = True
                elif p2_found and is_standoff:
                    self.standoff_action_idx = a_idx
                    self.entry_cell = end_cell
                    break

        self.planned = True

    def decide(self, s):
        dt = s["dt"]
        front_dist = min(s["sl"], s["sr"])
        fl, fr = s["fl"], s["fr"]
        yaw_rate = s["gyro"][2]
        tl, tr = s["ticks_l"], s["ticks_r"]
        self.sim_time = s.get("t", 0.0)

        theta_mid = normalize_angle(self.theta + 0.5 * yaw_rate * dt)
        self.theta = normalize_angle(self.theta + yaw_rate * dt)
        if self.prev_ticks is not None:
            dl = tl - self.prev_ticks[0]
            dr = tr - self.prev_ticks[1]
            ds = WHEEL_RADIUS * ((dl + dr) / 2.0) * TICK_RAD
            self.axle_x += ds * math.cos(theta_mid)
            self.axle_y += ds * math.sin(theta_mid)
            self.total_traveled += ds
        self.prev_ticks = (tl, tr)

        if not self.have_pellets or self._ghost is None:
            return 0.0, 0.0

        if not self.planned:
            self.init_frames += 1
            loop = extract_patrol_loop(self.ghost_history)
            if loop is None and self.init_frames < 2000:
                return 0.0, 0.0
            self.plan_route()
            self.state = "NEXT_ACTION"

        if self.action_idx >= len(self.actions):
            return 0.0, 0.0

        if self.state == "NEXT_ACTION":
            act = self.actions[self.action_idx]
            act_type = act[0]

            if act_type == "CHUTE":
                self.seg_start_dist = self.total_traveled
                self.state = "CHUTE"

            elif act_type == "TURN":
                self.brake_frames = 0
                self.state = "PRE_TURN_BRAKE"

            elif act_type == "DRIVE_SEG":
                req_h, end_cell, wall_ahead, is_standoff = act[1], act[2], act[3], act[4]
                if is_standoff and self._ghost and not self.standoff_cleared:
                    gx, gy = self._ghost["x"], self._ghost["y"]
                    ex = end_cell[1] * CELL_PITCH
                    ey = (14 - end_cell[0]) * CELL_PITCH
                    dist_to_entry = math.hypot(gx - ex, gy - ey)

                    if dist_to_entry <= 0.20:
                        self.standoff_ghost_was_close = True

                    is_moving_away = (dist_to_entry > self.last_dist_to_entry)
                    self.last_dist_to_entry = dist_to_entry

                    if (self.standoff_ghost_was_close and is_moving_away and dist_to_entry >= 0.35) or (dist_to_entry >= 0.60):
                        self.standoff_cleared = True
                    else:
                        return 0.0, 0.0

                self.seg_start_dist = self.total_traveled
                self.state = "DRIVE_SEG"

            elif act_type == "EXIT":
                self.state = "EXIT"

        # 1. PRE-TURN BRAKE
        if self.state == "PRE_TURN_BRAKE":
            self.brake_frames += 1
            if self.brake_frames >= 2 and abs(yaw_rate) < 0.05:
                c_col = int(round(self.axle_x / CELL_PITCH))
                c_row = 14 - int(round(self.axle_y / CELL_PITCH))
                c_row = max(0, min(14, c_row))
                c_col = max(0, min(14, c_col))
                cx = c_col * CELL_PITCH
                cy = (14 - c_row) * CELL_PITCH

                h = self.current_heading
                h_left = (h + 1) % 4
                h_right = (h + 3) % 4
                map_has_left = not self.maze.can_move((c_row, c_col), h_left)
                map_has_right = not self.maze.can_move((c_row, c_col), h_right)

                if map_has_left and map_has_right and 0.070 < fl < 0.130 and 0.070 < fr < 0.130 and abs((fl + fr) - 0.200) < 0.010:
                    e_lat = (fr - fl) / 2.0
                    if h == 0: self.axle_y = cy + e_lat
                    elif h == 2: self.axle_y = cy - e_lat
                    elif h == 1: self.axle_x = cx - e_lat
                    elif h == 3: self.axle_x = cx + e_lat

                if not self.maze.can_move((c_row, c_col), h):
                    avg_fwd = (s["sl"] + s["sr"]) / 2.0
                    if abs(s["sl"] - s["sr"]) < 0.015 and 0.065 < avg_fwd < 0.090:
                        e_long = avg_fwd - 0.077
                        if h == 0: self.axle_x = cx - e_long
                        elif h == 2: self.axle_x = cx + e_long
                        elif h == 1: self.axle_y = cy - e_long
                        elif h == 3: self.axle_y = cy + e_long

                self.state = "TURN"
            return 0.0, 0.0

        # 2. CHUTE DRIVE
        if self.state == "CHUTE":
            # Precision chute landing: approach slowly before snapping position
            if front_dist <= 0.298:
                self.axle_x = 0.00
                self.axle_y = ((fr - fl) / 2.0) if (fl < 0.15 and fr < 0.15) else 0.00
                self.theta = 0.0
                self.current_heading = 0
                self.action_idx += 1
                self.state = "NEXT_ACTION"
                return 0.0, 0.0

            fwd_speed = 4.5 if front_dist < 0.35 else 7.5
            e_lat = ((fr - fl) / 2.0) if (fl < 0.15 and fr < 0.15) else (self.axle_y - 0.00)
            heading_bias = max(-0.02, min(0.02, -0.6 * e_lat))
            target_angle = 0.0 + heading_bias
            turn_cmd = 2.0 * normalize_angle(target_angle - self.theta) - 0.05 * yaw_rate
            return fwd_speed - turn_cmd, fwd_speed + turn_cmd

        # 3. IN-PLACE TURN
        if self.state == "TURN":
            req_h = self.actions[self.action_idx][1]
            target_angle = CARDINAL_HEADINGS[req_h]
            angle_err = normalize_angle(target_angle - self.theta)

            if abs(angle_err) <= TURN_TOLERANCE:
                self.settle_frames = 0
                self.state = "POST_TURN_SETTLE"
                return 0.0, 0.0

            if abs(angle_err) > 0.35:
                turn_spd = 4.2
            elif abs(angle_err) > 0.08:
                turn_spd = 2.2
            else:
                turn_spd = max(0.8, min(1.4, 16.0 * abs(angle_err)))

            if angle_err > 0:
                return -turn_spd, turn_spd
            else:
                return turn_spd, -turn_spd

        # 4. POST-TURN SETTLE
        if self.state == "POST_TURN_SETTLE":
            self.settle_frames += 1
            if self.settle_frames >= 2 and abs(yaw_rate) < 0.05:
                req_h = self.actions[self.action_idx][1]
                self.current_heading = req_h
                self.action_idx += 1
                self.state = "NEXT_ACTION"
            return 0.0, 0.0

        # 5. DRIVE SEGMENT
        if self.state == "DRIVE_SEG":
            req_h, end_cell, wall_ahead, is_standoff = (
                self.actions[self.action_idx][1],
                self.actions[self.action_idx][2],
                self.actions[self.action_idx][3],
                self.actions[self.action_idx][4]
            )

            target_x = end_cell[1] * CELL_PITCH
            target_y = (14 - end_cell[0]) * CELL_PITCH

            if req_h == 0: rem_odom = target_x - self.axle_x
            elif req_h == 1: rem_odom = target_y - self.axle_y
            elif req_h == 2: rem_odom = self.axle_x - target_x
            else: rem_odom = self.axle_y - target_y

            avg_front = (s["sl"] + s["sr"]) / 2.0
            min_front = min(s["sl"], s["sr"])
            d_target = get_target_wall_dist(self.maze, end_cell, req_h)
            rem_laser = avg_front - d_target

            if wall_ahead and rem_odom <= 0.120 and min_front < 0.160:
                if min_front <= 0.080: remaining = 0.0
                elif abs(s["sl"] - s["sr"]) < 0.025: remaining = avg_front - 0.077
                else: remaining = min_front - 0.077
            elif rem_odom <= 0.100 and d_target < 1.60 and abs(s["sl"] - s["sr"]) < 0.025 and abs(rem_laser - rem_odom) < 0.040:
                remaining = rem_laser
            else:
                remaining = rem_odom

            if remaining <= 0.008 or remaining < 0.0:
                if req_h == 0 or req_h == 2: self.axle_x = target_x
                else: self.axle_y = target_y
                self.action_idx += 1
                self.state = "NEXT_ACTION"
                return 0.0, 0.0

            c_col = int(round(self.axle_x / CELL_PITCH))
            c_row = 14 - int(round(self.axle_y / CELL_PITCH))
            c_row = max(0, min(14, c_row))
            c_col = max(0, min(14, c_col))
            curr_c = (c_row, c_col)

            h_left = (req_h + 1) % 4
            h_right = (req_h + 3) % 4
            map_has_left = not self.maze.can_move(curr_c, h_left)
            map_has_right = not self.maze.can_move(curr_c, h_right)

            cardinal_angle = CARDINAL_HEADINGS[req_h]
            heading_err = normalize_angle(cardinal_angle - self.theta)

            is_dual_wall = (
                map_has_left and map_has_right and
                0.070 < fl < 0.130 and 0.070 < fr < 0.130 and
                abs((fl + fr) - 0.200) < 0.012 and
                abs(heading_err) < 0.04
            )

            if is_dual_wall:
                e_lat = (fr - fl) / 2.0
                lat_target = c_col * CELL_PITCH if (req_h in (1, 3)) else (14 - c_row) * CELL_PITCH
                if req_h == 0: self.axle_y = lat_target + e_lat
                elif req_h == 2: self.axle_y = lat_target - e_lat
                elif req_h == 1: self.axle_x = lat_target - e_lat
                else: self.axle_x = lat_target + e_lat

            # Active wall repulsion: push off side walls to prevent corridor scraping
            lat_bias = 0.0
            if map_has_left and fl < 0.068:
                lat_bias = 0.04
            elif map_has_right and fr < 0.068:
                lat_bias = -0.04

            target_angle = cardinal_angle + lat_bias

            dist_in_seg = self.total_traveled - self.seg_start_dist
            if dist_in_seg < 0.05:
                fwd_speed = START_SPEED + (CRUISE_SPEED - START_SPEED) * (dist_in_seg / 0.05)
            elif remaining <= 0.08:
                fwd_speed = APPROACH_SPEED
            else:
                fwd_speed = CRUISE_SPEED

            if self._ghost:
                gx, gy = self._ghost["x"], self._ghost["y"]
                dist_to_ghost = math.hypot(self.chassis_x - gx, self.chassis_y - gy)
                is_closing = (dist_to_ghost < getattr(self, "last_dist_to_ghost", float("inf")))
                self.last_dist_to_ghost = dist_to_ghost
                if dist_to_ghost < 0.20 or (dist_to_ghost < 0.28 and is_closing):
                    return 0.0, 0.0

            turn_cmd = 2.8 * normalize_angle(target_angle - self.theta) - 0.08 * yaw_rate
            turn_cmd = max(-1.0, min(1.0, turn_cmd))

            w_left = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, fwd_speed - turn_cmd))
            w_right = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, fwd_speed + turn_cmd))
            return w_left, w_right

        # 6. EXIT CHUTE
        if self.state == "EXIT":
            if self.chassis_x > 3.25:
                return 0.0, 0.0
            return 24.0, 24.0

        return 0.0, 0.0


Controller = Driver


def main():
    maze = Maze.official()
    print(f"official maze {maze_2b.MAZE_ID}: {maze.grid}x{maze.grid}, "
          f"spawn row {maze.spawn_row}, exit row {maze.exit_row}", flush=True)

    controller = Driver(maze)
    client = _mqtt_client()

    def on_message(cli, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == TOPIC_PELLETS:
                controller.on_pellets(payload)
                return
            if msg.topic == TOPIC_GHOST:
                controller.on_ghost(payload)
                return
            if msg.topic == TOPIC_RESULT:
                print(f"run over: solved={payload['solved']}, "
                      f"collisions={payload['collisions']}", flush=True)
                cli.loop_stop()
                cli.disconnect()
                return

            left, right = controller.decide(payload)
            left = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, float(left)))
            right = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, float(right)))
            cli.publish(TOPIC_WHEEL_VEL,
                        json.dumps({"left": left, "right": right}))
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr, flush=True)

    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    client.subscribe(TOPIC_PELLETS)
    client.subscribe(TOPIC_GHOST)
    client.subscribe(TOPIC_RESULT)
    client.subscribe(TOPIC_SENSORS)
    print("connected; waiting for the simulator", flush=True)
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
