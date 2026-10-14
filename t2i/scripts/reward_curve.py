"""Reproduce the paper's text-to-image reward-vs-compute curve.

Every saved checkpoint of the reported run is evaluated with the same harness as the tables, at the curve
protocol (NFE 50, CFG 2.5, 200 quality samples, HPSv2 only, diversity 32x16, seed 42). Each result is
appended to a JSONL file, and the curve plots HPSv2 against cumulative GPU-hours.

    python t2i/scripts/reward_curve.py t2i/configs/wtf_t2i.yaml --run exps/wtf_t2i
    python t2i/scripts/reward_curve.py t2i/configs/wtf_t2i.yaml --run exps/wtf_t2i --plot-only

GPU-hours on the x axis use the per-step cost measured for the reported run (15.087 s/step on 8xH100).
Pass --sec-per-step to use your own measurement.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run", required=True, help="training out_dir containing checkpoints/")
    ap.add_argument("--curve", default=None, help="JSONL to append to (default <run>/reward_curve.jsonl)")
    ap.add_argument("--every", type=int, default=10, help="evaluate checkpoints whose step is a multiple of this")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--sec-per-step", type=float, default=15.087)
    ap.add_argument("--n-gpus-train", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    curve = a.curve or os.path.join(a.run, "reward_curve.jsonl")

    done = set()
    if os.path.exists(curve):
        for ln in open(curve):
            try:
                done.add(json.loads(ln)["step"])
            except Exception:
                pass

    if not a.plot_only:
        for ck in sorted(glob.glob(os.path.join(a.run, "checkpoints", "*.pt"))):
            m = re.match(r"(\d+)\.pt$", os.path.basename(ck))
            if not m:
                continue
            step = int(m.group(1))
            if step % a.every or step in done:
                continue
            out = os.path.join(a.run, "curve_evals", "s%07d" % step)
            rc = subprocess.call([sys.executable, os.path.join(ROOT, "eval.py"), a.config, "--ckpt", ck,
                                  "--nfe", "50", "--num-quality", "200", "--rewards", "hps",
                                  "--gpus", str(a.gpus), "--output-dir", out])
            js = os.path.join(out, "eval_results_nfe50.json")
            if rc or not os.path.exists(js):
                print("  step %d: eval failed" % step)
                continue
            d = json.load(open(js))
            row = {"step": step, "hps": d["quality"]["hps"]["mean"],
                   "clip_div": d["diversity"]["clip_pairwise_l2sq"]["mean"],
                   "dreamsim": d["diversity"]["dreamsim_pairwise_l2sq"]["mean"],
                   "n": 200, "protocol": "NFE 50 cfg 2.5 n=200 div 32x16 seed 42"}
            with open(curve, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("  step %-5d hps %.5f" % (step, row["hps"]))

    if not os.path.exists(curve):
        raise SystemExit(f"no results in {curve}: evaluate the checkpoints first (run without --plot-only)")
    rows = sorted((json.loads(l) for l in open(curve)), key=lambda r: r["step"])
    gpuh = [r["step"] * a.sec_per_step * a.n_gpus_train / 3600.0 for r in rows]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(gpuh, [r["hps"] for r in rows], color="#0e7c66", lw=1.6, label="WTF")
    ax.set_xlabel("cumulative GPU-hours")
    ax.set_ylabel("HPSv2 (n=200, NFE 50)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    png = os.path.splitext(curve)[0] + ".png"
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    print("wrote", png, "(%d points)" % len(rows))


if __name__ == "__main__":
    main()
