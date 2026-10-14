#!/usr/bin/env python3
"""Evaluation entry point for both benchmarks, with the protocol of the paper fixed.

    python eval.py t2i/configs/wtf_t2i.yaml --ckpt exps/wtf_t2i/checkpoints/0001460.pt --nfe 4
    python eval.py t2i/configs/wtf_t2i.yaml --base --nfe 50
    python eval.py imagenet/configs/wtf_imagenet.yaml --ckpt exps/wtf_imagenet/checkpoints/0012500.pt --nfe 1

Text-to-image: 1000 samples from the 100 prompts of sample100_photo50_paint50.txt (HPSv2, PickScore,
ImageReward), 32 prompts x 16 samples for diversity, CFG 2.5, seed 42; writes eval_results_nfe<n>.json.
ImageNet-256: the first 32 classes x 16 samples (all metrics on the same 512 images), CFG 1.0, seed 42;
writes pc.json.
"""
import argparse, os, shlex, subprocess, sys

from run import ROOT, detect_lane, dig, load_config, missing_substrate

PROTOCOL = {
    "num_quality": 1000,
    "num_div_prompts": 32,
    "num_div_samples": 16,
    "seed": 42,
}


def default_output_dir(ckpt, nfe):
    """eval_runs/<run>_<checkpoint>_nfe<n>, where <run> is the folder above checkpoints/."""
    if ckpt is None:
        return os.path.join("eval_runs", f"base_nfe{nfe}")
    path = os.path.abspath(ckpt)
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(path)
    run = os.path.basename(os.path.dirname(parent)) if os.path.basename(parent) == "checkpoints" \
        else os.path.basename(parent)
    return os.path.join("eval_runs", f"{run}_{stem}_nfe{nfe}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ckpt", default=None, help="checkpoint to evaluate")
    ap.add_argument("--base", action="store_true", help="text-to-image only: evaluate the base model")
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--gpus", type=int, default=1, help="text-to-image only (default: 1)")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="text-to-image only: LoRA rank of the checkpoint (default: adaptation.lora_rank)")
    ap.add_argument("--num-quality", type=int, default=None,
                    help="text-to-image only. Default = table protocol (1000). The reward curve uses 200.")
    ap.add_argument("--rewards", default="hps,pickscore,imagereward",
                    help="text-to-image only. Table protocol scores all three; the reward curve uses hps.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    lane = detect_lane(a.config, cfg)
    if a.base and lane != "t2i":
        sys.exit("--base is text-to-image only")
    if (a.ckpt is None) == (not a.base):
        sys.exit("pass exactly one of --ckpt or --base")
    out = a.output_dir or default_output_dir(a.ckpt, a.nfe)

    if lane == "t2i":
        gpus = a.gpus
        argv = ["--ckpt-path", a.ckpt or "none",
                "--ckpt-base", os.path.expandvars(str(dig(cfg, "substrate.ckpt"))),
                "--config", os.path.expandvars(str(dig(cfg, "substrate.config"))),
                "--lora-rank", str(a.lora_rank or dig(cfg, "adaptation.lora_rank")),
                "--nfe", str(a.nfe), "--output-dir", out,
                "--num-quality", str(a.num_quality or PROTOCOL["num_quality"]),
                "--num-div-prompts", str(PROTOCOL["num_div_prompts"]),
                "--num-div-samples", str(PROTOCOL["num_div_samples"]),
                "--seed", str(PROTOCOL["seed"]),
                "--cfg-scale", "2.5",
                "--rewards", a.rewards]
        if a.base:
            argv.append("--no-lora")
        cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", str(gpus),
               os.path.join(ROOT, "t2i", "src", "scripts", "eval.py")] + argv
    else:
        argv = ["--ckpt", a.ckpt, "--num-steps", str(a.nfe), "--out", os.path.join(out, "pc.json"),
                "--reward", "hps", "--all-rewards",
                "--reward-classes", "0",
                "--div-classes", "first",
                "--n-groups", str(PROTOCOL["num_div_prompts"]),
                "--k-per-group", str(PROTOCOL["num_div_samples"]),
                "--cfg-scale", "1.0"]
        cmd = [sys.executable, os.path.join(ROOT, "imagenet", "scripts", "eval.py")] + argv

    print(" ".join(shlex.quote(c) for c in cmd))
    if lane == "t2i":
        missing = missing_substrate(cfg, lane)
        if missing:
            msg = f"missing substrate path: {', '.join(missing)}. See docs/SETUP.md."
            if not a.dry_run:
                sys.exit(msg)
            print("warning:", msg)
    if a.dry_run:
        return
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
