import numpy as np

r = 0.1651
B = 0.555
s = 0.85

corridor = {'x': (0, 6), 'y': (0, 2)}

init_boxes = [
    np.array([2.0, 1.0]),
    np.array([4.0, 1.0]),
    np.array([6.0, 1.0]),
]

push_targets = [
    np.array([2.0, -1.5]),
    np.array([4.0, -1.5]),
    np.array([6.0, -1.5]),
]


def skid_steer(wR, wL):
    v = r*s * (wR + wL) / 2
    w = r   * (wR - wL) / B
    return v, w


def step_toward(pos, theta, target, dt=0.05):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dist = np.hypot(dx, dy)
    ang  = np.arctan2(dy, dx)
    ang_err = (ang - theta + np.pi) % (2*np.pi) - np.pi
    wR = 8.0 + 5.0*ang_err
    wL = 8.0 - 5.0*ang_err
    v, w = skid_steer(wR, wL)
    pos   = pos + dt * np.array([v*np.cos(theta), v*np.sin(theta)])
    theta = theta + dt*w
    return pos, theta, dist


def run_husky_log():
    pos   = np.array([-1.5, 1.0])
    theta = 0.0
    boxes = [b.copy() for b in init_boxes]

    husky_path = [pos.copy()]
    box_paths  = [[b.copy()] for b in boxes]

    for i in range(3):
        approach = boxes[i] - np.array([0.6, 0.0])
        for _ in range(300):
            pos, theta, d = step_toward(pos, theta, approach)
            husky_path.append(pos.copy())
            for j in range(3):
                box_paths[j].append(boxes[j].copy())
            if d < 0.08:
                break

        for _ in range(300):
            pos, theta, d = step_toward(pos, theta, push_targets[i])
            t = max(0, 1 - d / np.linalg.norm(push_targets[i] - boxes[i]))
            boxes[i] = (1-t)*init_boxes[i] + t*push_targets[i]
            husky_path.append(pos.copy())
            for j in range(3):
                box_paths[j].append(boxes[j].copy())
            if d < 0.15:
                break

    cleared = sum(1 for b in boxes if not in_corridor(b))
    print(f"cajas despejadas: {cleared}/3")
    return {'husky_path': husky_path, 'box_paths': box_paths, 'boxes': boxes}


def in_corridor(bp):
    cx, cy = corridor['x'], corridor['y']
    return cx[0] <= bp[0] <= cx[1] and cy[0] <= bp[1] <= cy[1]


def run_husky(show=True):
    log = run_husky_log()
    cleared = sum(1 for b in log['boxes'] if not in_corridor(b))
    return cleared == 3