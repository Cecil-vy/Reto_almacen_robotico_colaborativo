from sim import simANYmal

PHASE_HUSKY    = "husky"
PHASE_ANYMAL   = "anymal"
PHASE_PUZZLEBOT = "puzzlebots"
PHASE_DONE     = "done"


def run_husky():
    print("Fase 1")
    return True   


def run_anymal(start=(0,0), goal=(11.0, 3.6)):
    print("Fase 2")
    
    log = simANYmal(start=start, goal=goal, show=False)

    final = (log['x'][-1], log['y'][-1])

    err   = ((final[0]-goal[0])**2 + (final[1]-goal[1])**2)**0.5

    print(f"  final pos: {final}, err: {err:.3f} m")
    return err < 0.15


def run_puzzlebots():
    print("Fase 3")
    return True


def main():
    state = PHASE_HUSKY

    while state != PHASE_DONE:

        if state == PHASE_HUSKY:
            ok = run_husky()
            state = PHASE_ANYMAL if ok else PHASE_HUSKY

        elif state == PHASE_ANYMAL:
            ok = run_anymal()
            state = PHASE_PUZZLEBOT if ok else PHASE_DONE   

        elif state == PHASE_PUZZLEBOT:
            ok = run_puzzlebots()
            state = PHASE_DONE

if __name__ == "__main__":
    main()