"""Streaming trainer: learn from photos one at a time, as if they were arriving live.

Photos are processed in capture order. For each one:

  1. Render it from its own camera *first*, before a single optimization step has touched
     it. The score from this render therefore says how well the model predicted a view it
     had never been shown - which is the number this project is really about.
  2. Then take a few optimization steps, mixing that newest photo with randomly chosen
     earlier ones so the model does not forget what it already learned.

This replaces src/poc/loop.py, which did the same thing but with a fixed set of
flat-coloured Gaussians that could never gain detail.

    python -m realtime_training.stream \\
        --data /srv2/menhart/rt3dgs/data/scene2/undistorted --strategy mcmc

One honest caveat, inherited from loop.py: the camera positions and the starting points
come from a COLMAP run over the *whole* sequence, done in advance. So the model begins
with rough knowledge of parts of the scene it has not been shown yet. A genuinely live
system would have to build that up as it goes.
"""
import argparse
import csv
import os
import time

import torch

from .colmap import load_scene, scene_name
from .common import require_cuda, save_comparison
from .dataset import Images
from .metrics import photometric_loss, psnr, ssim
from .model import Splats, means_lr_at
from .strategies import Densifier


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="colmap image_undistorter output dir")
    p.add_argument("--out", default=None,
                   help="output dir; defaults to out/<scene>/stream")
    p.add_argument("--steps-per-frame", type=int, default=30,
                   help="optimization steps taken per incoming photo")
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink the photos by this factor before training")
    p.add_argument("--strategy", choices=("default", "mcmc"), default="mcmc",
                   help="how Gaussians are added and removed; see strategies.py")
    p.add_argument("--cap-max", type=int, default=1_000_000,
                   help="maximum number of Gaussians (mcmc strategy only)")
    p.add_argument("--new-frame-prob", type=float, default=0.5,
                   help="chance a step uses the newest photo instead of an earlier one")
    p.add_argument("--lambda-dssim", type=float, default=0.2,
                   help="weight of the structural-similarity term; 0 = plain pixel error")
    p.add_argument("--sh-degree", type=int, default=3,
                   help="how view-dependent colours may be, 0 = flat colour per Gaussian")
    p.add_argument("--absgrad", action="store_true",
                   help="judge where to add detail using unsigned gradients")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after this many photos; 0 = use all of them")
    p.add_argument("--no-images", action="store_true",
                   help="skip writing the per-photo comparison images")
    p.add_argument("--cache-size", type=int, default=64,
                   help="how many decoded photos to keep in memory at once")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("out", scene_name(args.data), "stream")
    return args


def main(argv=None):
    args = parse_args(argv)
    dev = require_cuda()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    scene = load_scene(args.data)
    images = Images(scene, args.scale, dev, args.cache_size)
    n_frames = min(args.max_frames, len(scene)) if args.max_frames else len(scene)
    total_steps = n_frames * args.steps_per_frame

    splats = Splats.from_points(scene.points, scene.point_colors, scene.extent,
                                sh_degree=args.sh_degree, device=dev, absgrad=args.absgrad)
    dens = Densifier(args.strategy, scene.extent, total_steps,
                     cap_max=args.cap_max, absgrad=args.absgrad)
    dens.check(splats)

    sh_every = max(1, total_steps // 30)
    print(f"{dev} | {n_frames} frames at {images.width}x{images.height} "
          f"| {splats.n} init Gaussians | {args.steps_per_frame} steps/frame "
          f"| strategy {args.strategy}")
    if args.strategy == "default":
        print("  note: the 'default' strategy dims every Gaussian at once every so often, "
              "and stops\n        adding detail partway through. Both are a poor fit for an "
              "open-ended stream of\n        photos - expect dips in the scores below. "
              "Try --strategy mcmc.")

    rows = []
    step = 0
    t0 = time.perf_counter()
    for t in range(n_frames):
        # --- Render first: what does the model predict for a photo it has never seen?
        gt = images[t]
        with torch.no_grad():
            pred, _, _ = splats.render(images.viewmat(t), images.K,
                                       images.width, images.height)
            pred = pred.clamp(0, 1)
            pred_psnr = psnr(pred, gt)
            pred_ssim = ssim(pred, gt).item()
        if not args.no_images:
            save_comparison(os.path.join(args.out, f"{t:05d}.png"), gt, pred)

        # --- Learn: newest frame with probability p, otherwise a uniformly replayed
        #     earlier frame, so earlier views are not forgotten.
        for _ in range(args.steps_per_frame):
            splats.raise_sh_degree(step, sh_every)
            splats.set_means_lr(means_lr_at(step, total_steps, scene.extent))

            use_new = torch.rand(()).item() < args.new_frame_prob
            i = t if use_new else int(torch.randint(0, t + 1, ()).item())
            gt_i = images[i]
            out, _, info = splats.render(images.viewmat(i), images.K,
                                         images.width, images.height)
            loss, _, _ = photometric_loss(out, gt_i, args.lambda_dssim)
            loss = loss + dens.regularizer(splats)

            dens.pre_backward(splats, step, info)
            loss.backward()
            splats.step()
            splats.zero_grad()
            dens.post_backward(splats, step, info)
            step += 1

        rows.append((t, scene.names[t], pred_psnr, pred_ssim, splats.n))
        print(f"frame {t + 1:4d}/{n_frames} {scene.names[t]}: "
              f"predicted PSNR {pred_psnr:5.2f} dB | SSIM {pred_ssim:.4f} "
              f"| {splats.n:7d} gaussians | {time.perf_counter() - t0:7.1f}s")

    csv_path = os.path.join(args.out, "predicted_psnr.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "name", "predicted_psnr_db", "predicted_ssim", "n_gaussians"])
        w.writerows(rows)

    vals = [r[2] for r in rows]
    tail = vals[-max(1, len(vals) // 10):]  # last 10% of the stream
    print(f"predicted PSNR: mean {sum(vals) / len(vals):5.2f} dB "
          f"| final-decile mean {sum(tail) / len(tail):5.2f} dB "
          f"| min {min(vals):5.2f} | max {max(vals):5.2f}")

    torch.save(splats.state_dict(), os.path.join(args.out, "splats.pt"))
    print(f"wrote {csv_path} and {os.path.join(args.out, 'splats.pt')}")


if __name__ == "__main__":
    main()
