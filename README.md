# rt3dgs

Gaussian Splatting training experiments, aimed at learning a scene from images arriving
one at a time in capture order rather than from a fixed set available up front.

COLMAP recovers camera poses and a sparse point cloud. [gsplat](https://github.com/nerfstudio-project/gsplat)
rasterizes the Gaussians and manages densification. The training loops are this repo's.

Two trainers over one model:

| | |
|---|---|
| `realtime_training.train` | Offline. One random view per step, standard recipe. Establishes the quality a scene supports. |
| `realtime_training.stream` | Online. Renders each incoming view *before* training on it, so the logged score is novel-view prediction. The research question. |

## Requirements

CUDA. gsplat compiles its kernels on first use and has no CPU backend, so training runs on
a server with two A100 gpus. Without CUDA only COLMAP, data generation and ffmpeg are supported.

```bash
nix develop        # CUDA shell: torch + gsplat + nvcc
nix develop .#cpu  # if CUDA is not available; no gsplat
```

Both shells sync dependencies and put `src/` on `PYTHONPATH`.

```bash
python src/poc/smoke.py   # gsplat check: expects "device: cuda", non-zero grad norm
```

## Layout

```
src/realtime_training/   trainers
  colmap.py              COLMAP text-model reader, holdout split
  dataset.py             on-demand image loading, bounded cache
  model.py               Gaussian parameters, optimizers, the gsplat render call
  strategies.py          densification, both gsplat strategies behind one interface
  metrics.py             PSNR, SSIM, photometric loss
  train.py               offline training
  stream.py              live training on streamed data
src/data_generator/      synthetic multi-view generator with exact ground-truth poses
src/poc/                 original from-scratch rasterizer, superseded; smoke test lives here
colmap.sh                COLMAP pipeline
```

All gsplat calls are confined to `model.py` and `strategies.py`.

## Usage

Poses and point cloud, from `../data/<scene>/images/*.jpg`:

```bash
./colmap.sh ../data/<scene>              # USE_GPU=1 on an NVIDIA machine
```

Writes `<scene>/undistorted/{images,sparse}`, with `sparse/` converted to text. The
trainers do not read COLMAP's binary model. Check the registered-image count that
`model_analyzer` prints against the input count; a large shortfall means the
reconstruction fragmented.

Offline baseline, then streaming:

```bash
python -m realtime_training.train \
  --data /srv2/menhart/rt3dgs/data/scene2/undistorted --steps 30000

python -m realtime_training.stream \
  --data /srv2/menhart/rt3dgs/data/scene2/undistorted --strategy mcmc
```

Output defaults to `out/<scene>/offline/` and `out/<scene>/stream/`, derived from the data
path; override with `--out`. Both write side-by-side ground-truth/render images. Streaming
also writes `predicted_psnr.csv`.

Assemble a video:

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'out/<scene>/stream/*.png' \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" out/<scene>/stream.mp4
```

## Options

Shared by both trainers:

| Option | Default | |
|---|---|---|
| `--scale` | `1.0` | Image downscale. Lowers memory and time, raises the score — only compare runs at equal scale |
| `--strategy` | `default` / `mcmc` | Densification strategy; see below |
| `--cap-max` | `1000000` | Gaussian ceiling, `mcmc` only |
| `--lambda-dssim` | `0.2` | Weight of the SSIM term; `0` leaves plain L1, costing ~0.5–1 dB |
| `--sh-degree` | `3` | Spherical-harmonic degree for view-dependent colour; `0` is flat colour |
| `--absgrad` | off | Use unsigned screen-space gradients to place detail |

`train` adds `--steps`, `--holdout` (default 8, every Nth view excluded from training),
`--eval-every`. `stream` adds `--steps-per-frame`, `--new-frame-prob` (share of steps
spent on the newest view rather than a replayed earlier one), `--max-frames`.

## Densification strategies

| | |
|---|---|
| `default` | Original 3DGS adaptive density control: gradient-driven split/clone, opacity pruning, periodic opacity reset. Unbounded Gaussian count, refinement stops partway. Reproduces published numbers — use for the offline baseline. |
| `mcmc` | 3DGS-MCMC: fixed Gaussian budget, relocating faded Gaussians instead of adding new ones. Flat memory and frame time, no opacity-reset shock, no refinement cutoff. Preferred for streaming. |

Both are exposed on both trainers so the comparison can be measured.

## Conventions

Shared by COLMAP, gsplat and `src/data_generator/rgbd_synth.py`.

- view matrices are world-to-camera
- camera looks along `+z`, `+y` down (OpenCV)
- quaternions ordered `(w, x, y, z)`
- pixel centres at `+0.5`
- black background
- scales stored logarithmically, opacities as logits

## Notes

- **Gaussian count must grow** past the initial COLMAP point count. A flat count means
  densification is not receiving gradients.
- Offline holdout PSNR below ~25 dB means something is wrong; published 3DGS reaches
  27–31 dB. The superseded `src/poc` loop plateaued near 22 dB.
- `.envrc` selects a nix shell. If it loads the CPU shell on a CUDA machine, nothing
  errors. The trainers refuse to start; `src/poc` does not.
- Out of memory: lower `--scale` first. It does not bound the Gaussian count — use
  `--strategy mcmc --cap-max` for that.
- `src/data_generator/rgbd_synth.py` provides exact poses, useful for separating code bugs
  from bad reconstructions. Only three fixed viewpoints, so it needs extending to an orbit
  before it can train a meaningful model.

## Status

Trainers are written and verified against synthetic fixtures, but have not yet been run on
real scene data. The gsplat API calls in `model.py` and `strategies.py` are unverified
against the installed library.
