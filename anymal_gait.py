import numpy as np

class Leg:
    def __init__(self, name, side=1, l0=0.0585, l1=0.35, l2=0.33):
        self.name = name
        self.side = side
        self.l0, self.l1, self.l2 = l0, l1, l2
        self.q = np.zeros(3)

    def fk(self, q=None):
        if q is not None:
            self.q = np.asarray(q, float)
        q1, q2, q3 = self.q
        x =  self.l1*np.sin(q2) + self.l2* np.sin(q2 + q3)
        y =  self.side * self.l0*np.cos(q1)
        z = -self.l1*np.cos(q2) - self.l2 *np.cos(q2 + q3)
        return np.array([x, y, z])

    def ik(self, p):
        x, y, z = p
        ryz2 = y**2 + z**2 - self.l0**2
        ryz  = np.sqrt(max(ryz2, 1e-9))
        q1   = np.arctan2(y, -z) - np.arctan2(self.side*self.l0, ryz)
        D  = (x**2 + z**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)
        q3 = -np.arccos(np.clip(D, -1, 1))
        alpha = np.arctan2(x, -z)
        beta  = np.arctan2(self.l2*np.sin(-q3), self.l1 + self.l2*np.cos(q3))
        q2    = alpha - beta
        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        if q is None:
            q = self.q
        q1, q2, q3 = q
        J = np.zeros((3, 3))
        J[0,1] =  self.l1*np.cos(q2) + self.l2*np.cos(q2 + q3)
        J[0,2] =  self.l2*np.cos(q2 + q3)
        J[1,0] = -self.side*self.l0*np.sin(q1)
        J[2,1] =  self.l1*np.sin(q2) + self.l2*np.sin(q2 + q3)
        J[2,2] =  self.l2*np.sin(q2 + q3)
        return J

    def det_J(self, q=None):
        return np.linalg.det(self.jacobian(q))

    def is_singular(self, q=None, tol=1e-3):
        return abs(self.det_J(q)) < tol


class ANYmal:
    LEGS = ['LF', 'RF', 'LH', 'RH']

    def __init__(self):
        self.legs = {
            'LF': Leg('LF', side=+1),
            'RF': Leg('RF', side=-1),
            'LH': Leg('LH', side=+1),
            'RH': Leg('RH', side=-1),
        }
        self.pos   = np.array([0.0, 0.0])
        self.theta = 0.0

    def set_q(self, q12):
        q12 = np.asarray(q12)
        for i, name in enumerate(self.LEGS):
            self.legs[name].q = q12[3*i:3*i+3].copy()

    def get_q(self):
        return np.concatenate([self.legs[n].q for n in self.LEGS])

    def foot_positions(self):
        return {n: self.legs[n].fk() for n in self.LEGS}

    def check_singularities(self, tol=1e-3):
        return {n: self.legs[n].is_singular(tol=tol) for n in self.LEGS}


def plan_foot_traj(p_start, p_end, n_steps=20, clearance=0.08):
    pts = []
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        p = (1-t)*p_start + t*p_end
        p[2] += clearance * np.sin(np.pi * t)
        pts.append(p.copy())
    return pts


def trot_step(robot, step_len=0.15, dt=0.05):
    diag_a = ['LF', 'RH']
    diag_b = ['RF', 'LH']
    stance_q2, stance_q3 = 0.7, -1.4

    det_log = {}
    for name in robot.LEGS:
        leg = robot.legs[name]
        if name in diag_a:
            p_cur = leg.fk()
            p_tar = p_cur + np.array([step_len, 0, 0])
            traj  = plan_foot_traj(p_cur, p_tar)
            for p in traj:
                q_new = leg.ik(p)
                if not leg.is_singular(q_new):
                    leg.q = q_new
        else:
            leg.q[1] = stance_q2
            leg.q[2] = stance_q3
        det_log[name] = leg.det_J()

    robot.pos[0] += step_len * np.cos(robot.theta)
    robot.pos[1] += step_len * np.sin(robot.theta)
    return det_log


def simANYmal(start=(0.0, 0.0), goal=(11.0, 3.6), show=True):
    robot = ANYmal()
    robot.pos = np.array(start, float)

    for name in robot.LEGS:
        robot.legs[name].q = np.array([0.0, 0.7, -1.4])

    log = {'x': [robot.pos[0]], 'y': [robot.pos[1]], 'dets': []}

    max_steps = 300
    step_len  = 0.15

    for _ in range(max_steps):
        dx = goal[0] - robot.pos[0]
        dy = goal[1] - robot.pos[1]
        dist = np.hypot(dx, dy)
        if dist < 0.15:
            break
        robot.theta = np.arctan2(dy, dx)
        sl = min(step_len, dist)
        dets = trot_step(robot, step_len=sl)
        log['x'].append(robot.pos[0])
        log['y'].append(robot.pos[1])
        log['dets'].append(dets)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(log['x'], log['y'], 'b.-', markersize=3)
        plt.plot(*start, 'go', markersize=8)
        plt.plot(*goal,  'r*', markersize=12)
        plt.title('ANYmal trot path')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return log