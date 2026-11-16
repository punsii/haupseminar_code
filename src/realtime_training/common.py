"""Helpers shared by both trainers."""
import os

import imageio.v3 as iio
import torch

from .metrics import psnr, ssim


def require_cuda():
    """Stop immediately if there is no graphics card to train on.

    gsplat builds its code for the GPU the first time it runs and has no version that
    works on a plain processor, so there is nothing to fall back to. Worth failing loudly:
    the trap is that the nix shell chosen by `.envrc` may be the CPU one even on a machine
    with a graphics card, in which case training would quietly run thousands of times
    slower instead of complaining.
    """
    if not torch.cuda.is_available():
        raise SystemExit(
            "No graphics card available, and gsplat cannot run without one.\n"
            "On hmgpu use the default dev shell (`nix develop`), not `.#cpu`."
        )
    return "cuda"


def save_comparison(path, gt, pred):
    """Write the photo and the model's render of it next to each other, in one image.

    Same layout as src/poc/loop.py produced: real photo on the left, render on the right.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = torch.cat([gt, pred.clamp(0, 1)], dim=1)
    iio.imwrite(path, (img * 255).byte().cpu().numpy())


@torch.no_grad()
def evaluate(splats, images, idxs, out_dir=None, tag=""):
    """Average PSNR and SSIM over the given photos. Optionally writes comparison images."""
    if not idxs:
        return float("nan"), float("nan")
    ps, ss = [], []
    for k, i in enumerate(idxs):
        gt = images[i]
        pred, _, _ = splats.render(images.viewmat(i), images.K, images.width, images.height)
        pred = pred.clamp(0, 1)
        ps.append(psnr(pred, gt))
        ss.append(ssim(pred, gt).item())
        if out_dir:
            save_comparison(os.path.join(out_dir, f"{tag}{k:05d}.png"), gt, pred)
    return sum(ps) / len(ps), sum(ss) / len(ss)
