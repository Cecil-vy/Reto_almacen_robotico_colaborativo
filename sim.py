import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from anymal_gait   import simANYmal
from husky_pusher  import run_husky_log
from puzzlebot_arm import run_puzzlebots_log


def _wait_for_n(fig):
    fig.canvas.mpl_connect('key_press_event',
                           lambda e: plt.close(fig) if e.key == 'n' else None)
    fig.text(0.5, 0.01, 'presiona N para continuar', ha='center',
             fontsize=9, color='gray')


def sim_phase1():
    log       = run_husky_log()
    h_path    = log['husky_path']
    b_paths   = log['box_paths']
    n         = len(h_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    rect = mpatches.Rectangle((0, 0), 6, 2, color='lightyellow', label='corredor')
    ax.add_patch(rect)
    ax.set_xlim(-2, 10); ax.set_ylim(-3, 4)
    ax.set_title('Fase 1: Husky despejando corredor')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(True)

    h_dot,   = ax.plot([], [], 'bs', markersize=12, label='Husky')
    h_trail, = ax.plot([], [], 'b--', lw=1, alpha=0.4)

    cols     = ['red', 'orange', 'purple']
    b_dots   = [ax.plot([], [], 's', color=c, markersize=14,
                         label=f'Caja B{i+1}')[0] for i, c in enumerate(cols)]
    ax.legend(loc='upper right', fontsize=8)

    interval = 60

    def update(frame):
        i = min(frame, n - 1)
        hx, hy = h_path[i]
        h_dot.set_data([hx], [hy])
        h_trail.set_data([p[0] for p in h_path[:i+1]],
                          [p[1] for p in h_path[:i+1]])
        for j, bd in enumerate(b_dots):
            bi = min(frame, len(b_paths[j]) - 1)
            bd.set_data([b_paths[j][bi][0]], [b_paths[j][bi][1]])
        return [h_dot, h_trail] + b_dots

    anim = FuncAnimation(fig, update, frames=n, interval=interval,
                         blit=True, repeat=False)
    _wait_for_n(fig)
    plt.tight_layout()
    plt.show()


def sim_phase2():
    log = simANYmal(start=(0, 0), goal=(11.0, 3.6), show=False)
    xs, ys = log['x'], log['y']
    n      = len(xs)

    fig, ax = plt.subplots(figsize=(10, 5))
    rect = mpatches.Rectangle((0, 0), 6, 2, color='lightyellow', alpha=0.5,
                               label='corredor despejado')
    ax.add_patch(rect)
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='inicio')
    ax.plot(11.0, 3.6,    'r*', markersize=14, label='destino')
    ax.set_xlim(-1, 13); ax.set_ylim(-1, 6)
    ax.set_title('Fase 2: ANYmal transportando PuzzleBots')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(fontsize=8); ax.grid(True)

    trail,  = ax.plot([], [], 'b-', lw=2)
    body,   = ax.plot([], [], 'ko', markersize=12, label='ANYmal')
    pb_cols = ['cyan', 'magenta', 'lime']
    pb_dots = [ax.plot([], [], '^', color=c, markersize=7)[0] for c in pb_cols]
    offsets = [-0.15, 0.0, 0.15]
    interval = 40

    def update(frame):
        i = min(frame, n - 1)
        trail.set_data(xs[:i+1], ys[:i+1])
        body.set_data([xs[i]], [ys[i]])
        theta = np.arctan2(ys[min(i+1,n-1)]-ys[i],
                           xs[min(i+1,n-1)]-xs[i]) if i < n-1 else 0
        perp  = theta + np.pi/2
        for d, off in zip(pb_dots, offsets):
            d.set_data([xs[i] + off*np.cos(perp)],
                       [ys[i] + off*np.sin(perp)])
        return [trail, body] + pb_dots

    anim = FuncAnimation(fig, update, frames=n, interval=interval,
                         blit=True, repeat=False)
    _wait_for_n(fig)
    plt.tight_layout()
    plt.show()


def sim_phase3():
    log   = run_puzzlebots_log()
    paths = log['paths']      
    boxes = log['boxes']     
    bpths = log['box_paths']  
    stack = log['stack']
    n     = max(len(p) for p in paths.values())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(8, 12); ax.set_ylim(2, 5)
    ax.set_title('Fase 3: PuzzleBots apilando cajas')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(True)

    ax.plot(*stack, 'r^', markersize=14, label='pila destino')

    cols     = ['red', 'green', 'blue']
    box_cols = {'C': 'red', 'B': 'green', 'A': 'blue'}
    order    = ['C', 'B', 'A']

    bot_dots   = []
    bot_trails = []
    for i, c in enumerate(cols):
        d, = ax.plot([], [], 'o', color=c, markersize=10, label=f'PB{i}')
        t, = ax.plot([], [], '-', color=c, lw=1, alpha=0.5)
        bot_dots.append(d); bot_trails.append(t)

    box_dots = {}
    for bname in order:
        d, = ax.plot(*boxes[bname], 's', color=box_cols[bname],
                     markersize=12, label=f'Caja {bname}')
        box_dots[bname] = d

    cnt_txt = ax.text(8.1, 2.1, 'apiladas: 0/3', fontsize=9, color='darkred')
    ax.legend(fontsize=7, loc='upper right')
    interval = 30

    def update(frame):
        done = 0
        for i, (pid, path) in enumerate(paths.items()):
            fi = min(frame, len(path) - 1)
            px, py = path[fi]
            bot_dots[i].set_data([px], [py])
            bot_trails[i].set_data([p[0] for p in path[:fi+1]],
                                    [p[1] for p in path[:fi+1]])
            if fi == len(path) - 1:
                done += 1

        for bname in order:
            bp = bpths[bname]
            bi = min(frame, len(bp) - 1)
            box_dots[bname].set_data([bp[bi][0]], [bp[bi][1]])

        cnt_txt.set_text(f'apiladas: {done}/3')
        return bot_dots + bot_trails + list(box_dots.values()) + [cnt_txt]

    anim = FuncAnimation(fig, update, frames=n, interval=interval,
                         blit=True, repeat=False)
    _wait_for_n(fig)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim_phase1()
    sim_phase2()
    sim_phase3()