# Reference arguments

Saved training arguments of the reported runs, as written by the training scripts to `<out_dir>/args.json`.
Machine-specific paths are replaced by `<path>`.

| file | run |
|---|---|
| `t2i_args.json` | the text-to-image WTF run; its step-1460 checkpoint gives every WTF row of the text-to-image table (the run trained to 1,800 steps) |
| `imagenet_args_seed1.json` | ImageNet seed 1; the seed 0 and seed 2 runs differ only in `seed` |

A new run's `args.json` should match these on every key except paths, output and logging locations, and
the seed (see [`REPRODUCE.md`](../REPRODUCE.md)).
