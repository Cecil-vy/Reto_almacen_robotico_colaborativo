from husky_pusher  import run_husky
from anymal_gait   import simANYmal
from puzzlebot_arm import run_puzzlebots

PHASE_HUSKY     = "husky"
PHASE_ANYMAL    = "anymal"
PHASE_PUZZLEBOT = "puzzlebots"
PHASE_DONE      = "done"


def run_anymal(start=(0,0), goal=(11.0, 3.6)):
    log   = simANYmal(start=start, goal=goal, show=False)
    final = (log['x'][-1], log['y'][-1])
    err   = ((final[0]-goal[0])**2 + (final[1]-goal[1])**2)**0.5
    print(f"anymal final: {final}  err: {err:.3f} m")
    return err < 0.15


def main():
    state = PHASE_HUSKY
    while state != PHASE_DONE:
        if state == PHASE_HUSKY:
            ok    = run_husky(show=False)
            state = PHASE_ANYMAL if ok else PHASE_DONE
        elif state == PHASE_ANYMAL:
            ok    = run_anymal()
            state = PHASE_PUZZLEBOT if ok else PHASE_DONE
        elif state == PHASE_PUZZLEBOT:
            ok    = run_puzzlebots(show=False)
            state = PHASE_DONE

    print("done")


if __name__ == "__main__":
    main()