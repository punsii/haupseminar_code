"""Offline trainer: the ordinary way to train a Gaussian Splatting model.

Each step picks one random training photo, renders the model from that photo's camera,
and nudges the Gaussians to look more like it. Every 8th photo is set aside and never
trained on, so the quality score at the end is measured on views the model has not seen.

This trainer exists to prove the setup works. It follows the published recipe closely
enough that its score is comparable to published results, so if it scores well the
plumbing is sound and any disappointing result from stream.py is down to the streaming
schedule rather than a bug. Run it first on any new scene.

    python -m realtime_training.train \\
        --data /srv2/menhart/rt3dgs/data/scene2/undistorted --steps 30000
"""
import argparse
import os
import time

import torch

from .colmap import holdout_split, load_scene, scene_name
from .common import evaluate, require_cuda
from .dataset import Images
from .metrics import photometric_loss
from .model import Splats, means_lr_at
from .strategies import Densifier


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="colmap image_undistorter output dir")
    p.add_argument("--out", default=None,
                   help="output dir; defaults to out/<scene>/offline")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink the photos by this factor before training")
    p.add_argument("--strategy", choices=("default", "mcmc"), default="default",
                   help="how Gaussians are added and removed; see strategies.py")
    p.add_argument("--cap-max", type=int, default=1_000_000,
                   help="maximum number of Gaussians (mcmc strategy only)")
    p.add_argument("--holdout", type=int, default=8,
                   help="set aside every Nth photo for scoring, 0 = train on all of them")
    p.add_argument("--lambda-dssim", type=float, default=0.2,
                   help="weight of the structural-similarity term; 0 = plain pixel error")
    p.add_argument("--sh-degree", type=int, default=3,
                   help="how view-dependent colours may be, 0 = flat colour per Gaussian")
    p.add_argument("--absgrad", action="store_true",
                   help="judge where to add detail using unsigned gradients")
    p.add_argument("--eval-every", type=int, default=2000,
                   help="score the held-out photos this often, 0 = only at the end")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--cache-size", type=int, default=64,
                   help="how many decoded photos to keep in memory at once")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("out", scene_name(args.data), "offline")
    return args


def main(argv=None):
    args = parse_args(argv)
    dev = require_cuda()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    scene = load_scene(args.data)
    train_idx, test_idx = holdout_split(len(scene), args.holdout)
    images = Images(scene, args.scale, dev, args.cache_size)
    splats = Splats.from_points(scene.points, scene.point_colors, scene.extent,
                                sh_degree=args.sh_degree, device=dev, absgrad=args.absgrad)
    dens = Densifier(args.strategy, scene.extent, args.steps,
                     cap_max=args.cap_max, absgrad=args.absgrad)
    dens.check(splats)

    sh_every = max(1, args.steps // 30)  # reference raises one band per 1000 of 30k steps
    print(f"{dev} | {len(scene)} views ({len(train_idx)} train / {len(test_idx)} test) "
          f"at {images.width}x{images.height} | {splats.n} init Gaussians | "
          f"extent {scene.extent:.3f} | strategy {args.strategy}")

    t0 = time.perf_counter()
    for step in range(args.steps):
        splats.raise_sh_degree(step, sh_every)
        splats.set_means_lr(means_lr_at(step, args.steps, scene.extent))

        i = train_idx[int(torch.randint(len(train_idx), ()).item())]
        gt = images[i]
        pred, _, info = splats.render(images.viewmat(i), images.K,
                                      images.width, images.height)
        loss, l1, _ = photometric_loss(pred, gt, args.lambda_dssim)
        loss = loss + dens.regularizer(splats)

        # Order matters and follows gsplat's reference trainer: pre_backward retains the
        # means2d gradient, densification runs after the optimizer step.
        dens.pre_backward(splats, step, info)
        loss.backward()
        splats.step()
        splats.zero_grad()
        dens.post_backward(splats, step, info)

        if step % args.log_every == 0:
            print(f"step {step:6d}/{args.steps} | loss {loss.item():.4f} "
                  f"| L1 {l1.item():.4f} | {splats.n:7d} gaussians "
                  f"| sh {splats.active_sh_degree} | {time.perf_counter() - t0:7.1f}s")

        if args.eval_every and (step + 1) % args.eval_every == 0 and test_idx:
            ps, ss = evaluate(splats, images, test_idx)
            print(f"  [eval @ {step + 1}] holdout PSNR {ps:5.2f} dB | SSIM {ss:.4f} "
                  f"| {splats.n} gaussians")

    # Final evaluation, with images written out for inspection.
    ps, ss = evaluate(splats, images, test_idx, out_dir=os.path.join(args.out, "test"))
    tr_ps, tr_ss = evaluate(splats, images, train_idx[: len(test_idx) or 8])
    print(f"final | holdout PSNR {ps:5.2f} dB SSIM {ss:.4f} "
          f"| train-subset PSNR {tr_ps:5.2f} dB SSIM {tr_ss:.4f} | {splats.n} gaussians")

    torch.save(splats.state_dict(), os.path.join(args.out, "splats.pt"))
    print(f"wrote {os.path.join(args.out, 'splats.pt')}")


if __name__ == "__main__":
    main()
