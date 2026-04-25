import numpy as np


class PuzzleBotArm:
    def __init__(self, l1=0.10, l2=0.08, l3=0.06):
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.q = np.zeros(3)

    def fk(self, q=None):
        if q is not None:
            self.q = np.asarray(q, float)
        q1, q2, q3 = self.q
        l_eff = self.l2 + self.l3
        r = self.l1*np.cos(q2) + l_eff*np.cos(q2+q3)
        x = r*np.cos(q1)
        y = r*np.sin(q1)
        z = self.l1*np.sin(q2) + l_eff*np.sin(q2+q3)
        return np.array([x, y, z])

    def ik(self, p):
        x, y, z = p
        q1    = np.arctan2(y, x)
        r     = np.hypot(x, y)
        l_eff = self.l2 + self.l3
        D     = (r**2 + z**2 - self.l1**2 - l_eff**2) / (2*self.l1*l_eff)
        q3    = np.arccos(np.clip(D, -1, 1))
        alpha = np.arctan2(z, r)
        beta  = np.arctan2(l_eff*np.sin(q3), self.l1 + l_eff*np.cos(q3))
        q2    = alpha - beta
        self.q = np.array([q1, q2, q3])
        return self.q

    def jacobian(self, q=None):
        if q is not None:
            self.q = np.asarray(q, float)
        q1, q2, q3 = self.q
        l_eff = self.l2 + self.l3
        J = np.zeros((3, 3))
        r = self.l1*np.cos(q2) + l_eff*np.cos(q2+q3)
        J[0,0] = -r*np.sin(q1)
        J[1,0] =  r*np.cos(q1)
        J[0,1] = np.cos(q1)*(-self.l1*np.sin(q2) - l_eff*np.sin(q2+q3))
        J[1,1] = np.sin(q1)*(-self.l1*np.sin(q2) - l_eff*np.sin(q2+q3))
        J[2,1] =  self.l1*np.cos(q2) + l_eff*np.cos(q2+q3)
        J[0,2] = np.cos(q1)*(-l_eff*np.sin(q2+q3))
        J[1,2] = np.sin(q1)*(-l_eff*np.sin(q2+q3))
        J[2,2] =  l_eff*np.cos(q2+q3)
        return J

    def force_to_torque(self, f_tip):
        return self.jacobian().T @ f_tip

    def grasp_box(self, box_pos, grip_force=5.0):
        self.ik(np.array(box_pos) + np.array([0, 0, 0.05]))
        self.ik(box_pos)
        return self.force_to_torque(np.array([0, 0, -grip_force]))


class PuzzleBot:
    def __init__(self, pid, start_pos):
        self.pid = pid
        self.pos = np.array(start_pos, float)
        self.arm = PuzzleBotArm()

    def move_to(self, target, spd=0.3, dt=0.05):
        target = np.array(target, float)
        path   = [self.pos.copy()]
        for _ in range(800):
            d = np.linalg.norm(target - self.pos)
            if d < 0.04:
                break
            self.pos += dt*spd * (target - self.pos) / d
            path.append(self.pos.copy())
        return path


def run_puzzlebots_log():
    box_positions = {
        'A': np.array([9.0,  3.0]),
        'B': np.array([9.5,  3.0]),
        'C': np.array([10.0, 3.0]),
    }
    stack = np.array([10.5, 3.6])

    bots = [
        PuzzleBot(0, [11.0, 3.0]),
        PuzzleBot(1, [11.0, 3.3]),
        PuzzleBot(2, [11.0, 3.6]),
    ]

    order     = ['C', 'B', 'A']
    paths     = {b.pid: [] for b in bots}
    box_paths = {name: [] for name in order}

    for slot, (bot, box_name) in enumerate(zip(bots, order)):
        bp           = box_positions[box_name].copy()
        stack_target = stack + np.array([0, slot*0.05])

        # bot va a la caja — caja quieta
        seg1 = bot.move_to(bp)
        paths[bot.pid] += seg1
        box_paths[box_name] += [bp.copy()] * len(seg1)

        bot.arm.grasp_box(np.array([0.05, 0.0, 0.1]))

        # bot lleva la caja al stack — caja sigue al bot
        seg2 = bot.move_to(stack_target)
        paths[bot.pid] += seg2
        box_paths[box_name] += [np.array(p) for p in seg2]

        # bot regresa — caja se queda en stack
        seg3 = bot.move_to([11.0, 3.0 + slot*0.3])
        paths[bot.pid] += seg3
        box_paths[box_name] += [seg2[-1].copy()] * len(seg3)

    n = max(len(p) for p in paths.values())
    for pid in paths:
        last = paths[pid][-1]
        while len(paths[pid]) < n:
            paths[pid].append(last)
    for bname in box_paths:
        last = box_paths[bname][-1]
        while len(box_paths[bname]) < n:
            box_paths[bname].append(last)

    print("apilado completado C-B-A")
    return {
        'paths':     paths,
        'boxes':     box_positions,
        'box_paths': box_paths,
        'stack':     stack,
    }


def run_puzzlebots(show=True):
    run_puzzlebots_log()
    return True