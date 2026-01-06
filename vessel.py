import numpy as np
import random
import cv2
import utils
import math
from shapely.geometry import LineString, Point
from scipy.signal import savgol_filter

# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def build_path(path_xy):
    line = LineString(path_xy)
    seg_lengths = np.sqrt(np.sum(np.diff(path_xy, axis=0)**2, axis=1))
    cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
    return line, cumlen

def project_to_path(x, y, line, cumlen, path_xy):
    p = Point(x, y)

    dist_along = line.project(p)
    nearest_point = line.interpolate(dist_along)
    px, py = nearest_point.x, nearest_point.y

    seg_idx = np.searchsorted(cumlen, dist_along) - 1
    seg_idx = np.clip(seg_idx, 0, len(path_xy) - 2)

    p1 = path_xy[seg_idx]
    p2 = path_xy[seg_idx + 1]
    tangent = p2 - p1
    tangent = tangent / np.linalg.norm(tangent)
    tx, ty = tangent

    nx, ny = -ty, tx  # left-normal
    cross_track = (x - px) * nx + (y - py) * ny

    path_heading = math.atan2(ty, tx)

    return {
        "px": px,
        "py": py,
        "cross_track": cross_track,
        "path_heading": path_heading,
        "progress": dist_along,
    }


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=1.0, goal_sample_rate=5, max_iter=1000):
        self.start = Node(*start)
        self.end = Node(*goal)
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

    def planning(self):
        for _ in range(self.max_iter):
            rnd = self.sample_free()
            nearest_ind = self.get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]

            theta = np.arctan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = Node(nearest_node.x + self.expand_dis * np.cos(theta),
                            nearest_node.y + self.expand_dis * np.sin(theta))
            new_node.parent = nearest_node

            if not self.collision_check(new_node):
                continue

            self.node_list.append(new_node)

            if np.hypot(new_node.x - self.end.x, new_node.y - self.end.y) < self.expand_dis:
                return self.extract_path(new_node)

        return None

    def sample_free(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            return [np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(self.min_rand, self.max_rand)]
        return [self.end.x, self.end.y]

    def get_nearest_node_index(self, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in self.node_list]
        return int(np.argmin(dlist))

    def collision_check(self, node):
        for (ox, oy, size) in self.obstacle_list:
            if (ox - node.x)**2 + (oy - node.y)**2 <= size**2:
                return False
        return True

    def extract_path(self, goal_node):
        path = [[goal_node.x, goal_node.y]]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1]

class Vessel(object):

    def __init__(self, max_steps, task='navigating', vessel_type='vessel',
                 viewport_h=768, path_to_bg_img=None):

        self.task = task
        self.vessel_type = vessel_type

        self.g = 9.8
        self.L = 38. # vessel length (meters)
        self.B = 16. # vessel beam
        self.I = 1/12*self.L*self.L
        self.dt = 0.25

        self.world_x_min = -300  # meters
        self.world_x_max =  300
        self.world_y_min = -300
        self.world_y_max =  300

        # target point
        if self.task == 'navigating':
            self.target_x, self.target_y, self.target_r = 0, 0, 0.5*self.B

        self.already_achieved = False
        self.max_steps = max_steps

        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.dist_goal0 = 0
        self.goal = []
        self.obstacle_list = []
        self.rrt = None
        self.path = None
        self.dist_path = 0
        self.prev_progress = 1.

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        self.state_dims = 10
        self.action_dims = len(self.action_table)

        self.bg_img = utils.load_bg_img('ocean.jpg', w=self.viewport_w, h=self.viewport_h)

        self.state_buffer = []

        # wind model parameters
        self.rho_air = 1.225
        self.A_wind = self.L * self.B * 0.1
        self.Cd_wind = 1.0
        self.Cn_wind = 0.1
        self.T_wind = 25.0
        self.sigma_wind = 0.0  # wind intensity [m/s]
        self.wind_vel = np.zeros(2)

    def update_wind(self):
        dt = self.dt
        Tw = self.T_wind
        sigma = self.sigma_wind

        dw = np.random.randn(2)
        self.wind_vel += (
                -self.wind_vel / Tw * dt
                + sigma * np.sqrt(2.0 / Tw) * np.sqrt(dt) * dw
        )

        return self.wind_vel

    def wind_forces(self, u, v, psi):
        Vw = self.update_wind()

        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, s],
                      [-s, c]])

        V_rel = R @ Vw - np.array([u, v])
        V_mag = np.linalg.norm(V_rel) + 1e-6

        Fx = 0.5 * self.rho_air * self.A_wind * self.Cd_wind * V_mag * V_rel[0]
        Fy = 0.5 * self.rho_air * self.A_wind * self.Cd_wind * V_mag * V_rel[1]

        N = 0.5 * self.rho_air * self.A_wind * self.L * self.Cn_wind * V_mag * V_rel[1]

        return np.array([Fx, 0, 0])

    def smooth_path_savgol(self, path, window=30, poly=3, out_points=50):
        path = np.asarray(path)

        x = savgol_filter(path[:, 0], window, poly, mode='nearest')
        y = savgol_filter(path[:, 1], window, poly, mode='nearest')
        smoothed = np.column_stack((x, y))

        idx = np.linspace(0, len(smoothed) - 1, out_points).astype(int)
        idx = idx.tolist()
        idx.insert(0, 0)
        idx.append(len(smoothed) - 1)
        idx = np.asarray(idx)

        smoothed = smoothed[idx].tolist()
        smoothed.insert(0, path[0])
        smoothed.append(path[-1])

        return np.asarray(smoothed)


    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        self.state_buffer = []
        self.step_id = 0
        self.already_achieved = False

        self.goal = np.array([0, 0])
        xy = np.array([self.state['x'], self.state['y']], dtype=float)
        self.dist_goal0 = np.linalg.norm(xy - self.goal)
        self.obstacle_list = [
            [
                random.uniform(self.world_x_min*0.25, self.world_x_max*0.25),
                random.uniform(0, self.world_y_max),
                random.uniform(5, 50)
            ]
            for _ in range(3)
        ]
        self.rrt = RRT(start=xy,
                       goal=self.goal,
                       obstacle_list=self.obstacle_list,
                       rand_area=(-300, 300),
                       expand_dis=2.,
                       max_iter=10000)
        self.path = self.rrt.planning()
        self.path = np.asarray(self.path, dtype=float)
        self.path = self.smooth_path_savgol(self.path)

        cv2.destroyAllWindows()
        return self.flatten(self.state)

    def create_action_table(self):
        forces = [-1000, 1000]  # N
        moments = [0]  # N·m (yaw)
        action_table = [[X, Y, N] for X in forces for Y in forces for N in moments]
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0

        psi = 0

        x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
        y = yc + 0.4*y_range

        state = {
            'x': x,
            'y': y,
            'psi': psi,
            'u': 0,
            'v': 0,
            'r': 0,
            'dist_target': 0,
            'dist_pathx':0 ,
            'dist_pathy': 0,
            't': 0,
            'a_': 0
        }

        return state

    def check_navigating_success(self, state):
        x, y, psi = state['x'], state['y'], state['psi']
        if np.linalg.norm([x, y])<(0.5*self.L):
            return True
        else:
            return False

    def check_speed(self, u, v):
        if abs(u)>2.0 and abs(v)>2.0:
            return 2.0
        elif abs(u)>2.0 or abs(v)>2.0:
            return 1.0
        else:
            return 0.0

    def reward_dense(
            self,
            state,
            action,
            t,
            next_state,
            line,
            cumlen,
            path_xy,
            prev_progress,
            goal_point=(0.0, 0.0),
            reach_thresh=5.0,
            w_progress=1.0,
            w_cross=2.0,
            w_heading=0.2,
            w_act=0.01,
            w_time=0.001,
            w_speed=1.0
    ):

        x, y, theta, u, v = state[:5]
        nx, ny, ntheta = next_state[:3]

        info = project_to_path(nx, ny, line, cumlen, path_xy)

        cross_track = info["cross_track"]
        progress = info["progress"]
        delta_progress = progress - prev_progress

        delta_progress = max(delta_progress, -0.05)

        r = 0.0
        r += w_progress * delta_progress

        if abs(cross_track)>0.5:
            r -= w_cross * abs(cross_track)

        r -= w_speed * self.check_speed(u, v)
        r -= w_time

        if math.hypot(nx - goal_point[0], ny - goal_point[1]) <= reach_thresh:
            r += 10.0
            done = True
        else:
            done = False

        return r, progress, done, cross_track, delta_progress

    def check_time_and_distance(self, state):
        t, x, y, psi = state['t'], state['x'], state['y'], state['psi']
        percentage_path = (np.linalg.norm([x, y])/self.dist_goal0)
        if   (t > 100 ) and (percentage_path > 0.9):
            return True
        else:
            return False

    def calculate_reward(self, state, action, next_state):

        line, cumlen = build_path(self.path)

        total_reward, new_progress, done, cross_track, delta_progress = self.reward_dense(np.array([state['x'], state['y'], state['psi'], state['u'], state['v']], dtype=float),
                                    action,
                                    state['t'],
                                    next_state,
                                    line,
                                    cumlen,
                                    self.path,
                                    self.prev_progress,
                                    goal_point=(0.0, 0.0),
                                    reach_thresh=5.00,
                                    w_progress=1.0,
                                    w_cross=0.1,
                                    w_heading=0.2,
                                    w_act=0.01,
                                    w_time=0.001,
                                    w_speed=1.0)

        self.prev_progress = new_progress

        return total_reward, new_progress, cross_track, delta_progress

    def step(self, action):
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        u, v, r = self.state['u'], self.state['v'], self.state['r']

        tau = np.array(self.action_table[action])

        m = 500.0  # mass (kg)
        Iz = 200.0  # yaw inertia (kg·m²)
        X_u = -50.0  # surge damping
        Y_v = -200.0  # sway damping
        N_r = -100.0  # yaw damping

        M = np.diag([m, m, Iz])

        C = np.array([[0, -m * r, 0],
                      [m * r, 0, 0],
                      [0, 0, 0]])

        D = np.diag([-X_u, -Y_v, -N_r])

        nu = np.array([u, v, r])

        tau_wind = self.wind_forces(u, v, psi)
        tau_total = tau + tau_wind
        nu_dot = np.linalg.inv(M) @ (tau_total - (C + D) @ nu)

        nu_new = nu + nu_dot * self.dt
        u_new, v_new, r_new = nu_new

        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
        eta_dot = R @ nu
        eta_new = np.array([x, y, psi]) + eta_dot * self.dt
        x_new, y_new, psi_new = eta_new

        self.state = {
            'x': x_new,
            'y': y_new,
            'psi': psi_new,
            'u': u_new,
            'v': v_new,
            'r': r_new,
            'dist_target': 0,
            'dist_pathx':0 ,
            'dist_pathy': 0,
            't': self.step_id,
            'action_': action
        }
        self.step_id += 1

        done = False
        next_state = np.array([x_new, y_new, psi_new], dtype=float)
        reward, dist_target, dist_pathx, dist_pathy  = self.calculate_reward(self.state, action, next_state)
        self.state['dist_target'] = dist_target
        self.state['dist_pathx'] = dist_pathx
        self.state['dist_pathy'] = dist_pathy
        self.state_buffer.append(self.state)
        self.already_achieved = self.check_navigating_success(self.state)

        if self.already_achieved: #or self.check_time_and_distance(self.state):
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, None

    def flatten(self, state):
        x = [state['x'],
             state['y'],
             state['psi'],
             state['u'],
             state['v'],
             state['r'],
             state['dist_target'],
             state['dist_pathx'],
             state['dist_pathy'],
             state['t']]
        return np.array(x, dtype=np.float16)

    def render(self, window_name='env', wait_time=1,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()

        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        for poly in polys['vessel']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        if hasattr(self, 'path') and self.path is not None:
            path_px = self.wd2pxl(np.array(self.path))
            cv2.polylines(canvas, [path_px], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        if hasattr(self, 'obstacle_list') and len(self.obstacle_list) > 0:
            for (ox, oy, radius) in self.obstacle_list:
                center_px = self.wd2pxl(np.array([[ox, oy]]))[0]

                scale = self.viewport_w / (self.world_x_max - self.world_x_min)
                radius_px = int(radius * scale)

                cv2.circle(canvas, tuple(center_px), radius_px, (128, 128, 128), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, tuple(center_px), radius_px, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        cv2.imshow(window_name, frame_0[:,:,::-1])
        cv2.waitKey(wait_time)
        cv2.imshow(window_name, frame_1[:,:,::-1])
        cv2.waitKey(wait_time)
        return frame_0, frame_1

    def create_polygons(self):

        polys = {'vessel': [], 'engine_work': [], 'target_region': []}

        if self.vessel_type == 'vessel':

            L, B = self.L, self.B
            dl = self.L / 30

            Lb = 0.2 * L
            pts = [
                [-B / 2, L / 2],  # port stern
                [B / 2, L / 2],  # starboard stern
                [B / 2, -L / 2],  # starboard bow shoulder
                [0.0, -L / 2 - Lb],  # bow tip (downward)
                [-B / 2, -L / 2],  # port bow shoulder
            ]
            polys['vessel'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            f, psi = self.state['action_'], self.state['psi']
            c, s = np.cos(psi), np.sin(psi)

            if self.action_table[f][0] > 0:
                pts1 = utils.create_rectangle_poly(center=(B/2, L/2+5), w=1, h=5)
                pts2 = utils.create_rectangle_poly(center=(-B/2, L/2+5), w=1, h=5)
                polys['engine_work'].append({'pts': pts1, 'face_color': (0, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (0, 255, 255), 'edge_color': None})

            if self.action_table[f][0] < 0:
                pts1 = utils.create_rectangle_poly(center=(B / 2, -L / 2-5), w=1, h=5)
                pts2 = utils.create_rectangle_poly(center=(-B / 2, -L / 2-5), w=1, h=5)
                polys['engine_work'].append({'pts': pts1, 'face_color': (0, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (0, 255, 255), 'edge_color': None})

            if self.action_table[f][1] > 0:
                pts1 = utils.create_rectangle_poly(center=(B/2+5, L/2), w=5, h=1)
                pts2 = utils.create_rectangle_poly(center=(B/2+5, -L / 2), w=5, h=1)
                polys['engine_work'].append({'pts': pts1, 'face_color': (0, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (0, 255, 255), 'edge_color': None})

            if self.action_table[f][1] < 0:
                pts1 = utils.create_rectangle_poly(center=(-B/2+5-10, L/2), w=5, h=1)
                pts2 = utils.create_rectangle_poly(center=(-B/2+5-10, -L / 2), w=5, h=1)
                polys['engine_work'].append({'pts': pts1, 'face_color': (0, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (0, 255, 255), 'edge_color': None})


        pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r)
        polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})

        for poly in polys['vessel'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['psi'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state['x'], self.state['y'])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "u: %.2f m/s, v: %.2f m/s" % \
               (self.state['u'], self.state['v'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
               (self.state['psi'] * 180 / np.pi, self.state['r'] * 180 / np.pi)
        put_text(canvas, text, pt)


    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.L/2], [self.world_x_max - range_x/3, self.L/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        pts = [[0, self.L/2], [0, self.L/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis

