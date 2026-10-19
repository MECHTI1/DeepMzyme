import json, os, shlex, subprocess, sys, time
from pathlib import Path

cmds = json.loads(Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/execution/phase2_unified_runs/unified_commands.json").read_text())
out_dir = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/execution/phase2_unified_runs")
out_dir.mkdir(parents=True, exist_ok=True)
status_file = out_dir / "wave_status.json"

def run_wave(folds):
    procs = {}
    for f in folds:
        spec = cmds[str(f)]
        run_name = spec["run_name"]
        run_dir = out_dir / run_name
        if (run_dir / "terminal_model_checkpoint.pt").is_file() and (run_dir / "val_predictions.csv").is_file():
            print(f"[SKIP] {run_name} already completed", flush=True)
            continue
        env = dict(os.environ, **spec["env"])
        log_f = (out_dir / f"{run_name}.driver.log").open("w")
        print(f"[START] {run_name} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        p = subprocess.Popen(spec["argv"], env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd="/media/mechti/Data1/DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/code")
        procs[f] = (run_name, p, log_f, time.time())
    while procs:
        for f, (run_name, p, log_f, t0) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                log_f.close()
                dt = time.time() - t0
                print(f"[DONE] {run_name} rc={rc} elapsed={dt:.1f}s", flush=True)
                del procs[f]
        time.sleep(5)

print("=== WAVE 1: Folds 0, 1, 2 ===", flush=True)
run_wave([0, 1, 2])
print("=== WAVE 2: Folds 3, 4 ===", flush=True)
run_wave([3, 4])
print("=== ALL WAVES FINISHED ===", flush=True)
