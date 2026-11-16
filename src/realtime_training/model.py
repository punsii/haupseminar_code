"""The model itself: what each Gaussian stores, how it is optimized, and how it is drawn.

A scene is a big pile of 3D Gaussians - fuzzy, stretched blobs. Each one stores a
position, a size along three axes, an orientation, a transparency, and a colour. Training
adjusts all five. Two of them are stored transformed rather than directly: sizes are kept
as logarithms and transparency as a value that gets squashed into 0..1 later, both so that
the optimizer cannot push them into impossible values like a negative size.

Colour is not a single value. Real surfaces change appearance with viewing angle, so each
Gaussian stores a small set of coefficients that describe colour *as a function of
direction*. These use spherical harmonics, a standard set of basis functions on the
sphere, ordered from coarse to fine; `sh_degree` chooses how many of them to keep.
Degree 0 is one coefficient, meaning flat colour from every angle; degree 3 is sixteen,
enough for soft shine and sheen.

Two details of the layout here are forced by gsplat and are not free choices:

1. gsplat's population editing (see strategies.py) adds and deletes Gaussians mid-run,
   which means every tensor changes length. The optimizer keeps its own running averages
   per value and those have to be cut to the same length at the same moment. gsplat does
   that for us, but only if given a dictionary of parameters and a *separate optimizer
   per parameter* under the same keys. The single optimizer used in src/poc/loop.py
   cannot work.
2. Drawing returns a third value, a bag of internal quantities, that src/poc/render.py
   discarded. Population editing depends on it, so render() passes it back.

Everything that touches the gsplat library lives in this file and strategies.py, so if a
library function has changed shape, those are the only two places to correct.
"""
import math

import torch
from gsplat import rasterization

SH_C0 = 0.28209479177387814  # Y_00, the DC spherical-harmonic basis value

# 3DGS reference learning rates. `means` is additionally scaled by the scene extent.
LRS = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
}


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def knn_scales(xyz, k=3, budget_bytes=1 << 30):
    """Average distance from each point to its k closest neighbours.

    Used to pick a sensible starting size for each Gaussian: dense regions get small
    blobs, sparse regions get large ones.

    The distances are computed in blocks of rows. loop.py used a fixed block of 4096,
    which builds a 4096-by-N table of distances - 16 GB at a million points, and one
    cause of the earlier out-of-memory crash. Here the block is sized to fit a memory
    budget instead.
    """
    n = len(xyz)
    chunk = max(1, min(n, int(budget_bytes // (4 * max(n, 1)))))
    out = []
    for c in xyz.split(chunk):
        d = torch.cdist(c, xyz)  # [chunk, N]
        # topk includes the point itself at distance 0, so take k+1 and drop the first.
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))
    return torch.cat(out)


class Splats:
    def __init__(self, params, opts, extent, sh_degree=3, packed=False, absgrad=False):
        self.params = params
        self.opts = opts
        self.extent = extent
        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.packed = packed
        self.absgrad = absgrad

    @classmethod
    def from_points(cls, xyz, rgb, extent, sh_degree=3, init_opacity=0.1,
                    device="cuda", packed=False, absgrad=False):
        xyz = xyz.to(device).float()
        rgb = rgb.to(device).float().clamp(0, 1)
        n = len(xyz)
        bands = (sh_degree + 1) ** 2

        dist = knn_scales(xyz).clamp_min(1e-7)
        sh0 = rgb_to_sh(rgb)[:, None, :]  # [N, 1, 3] DC term
        shN = torch.zeros(n, bands - 1, 3, device=device)

        init = {
            "means": xyz,
            "scales": dist.log()[:, None].repeat(1, 3),  # log-space, isotropic
            "quats": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1),
            "opacities": torch.full((n,), init_opacity, device=device).logit(),
            "sh0": sh0,
            "shN": shN,
        }
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in init.items()})
        opts = {
            k: torch.optim.Adam([params[k]],
                                lr=LRS[k] * (extent if k == "means" else 1.0),
                                eps=1e-15)
            for k in init
        }
        return cls(params, opts, extent, sh_degree, packed, absgrad)

    # --- state -------------------------------------------------------------------
    @property
    def n(self):
        return self.params["means"].shape[0]

    def means_lr(self):
        """Current means learning rate. MCMCStrategy needs it for its noise term."""
        return self.opts["means"].param_groups[0]["lr"]

    def set_means_lr(self, lr):
        self.opts["means"].param_groups[0]["lr"] = lr

    def raise_sh_degree(self, step, every=1000):
        """3DGS warms up SH bands instead of optimizing them all from step 0."""
        target = min(step // every, self.sh_degree)
        self.active_sh_degree = target

    # --- render ------------------------------------------------------------------
    def render(self, viewmat, K, width, height):
        """One view. Returns rgb [H, W, 3], alpha [H, W, 1], info dict."""
        p = self.params
        # Hand over exactly as many colour coefficients as are currently switched on.
        # Two reasons: gsplat may insist the count matches the requested degree exactly,
        # and leaving the rest out keeps them untouched until their turn comes.
        bands = (self.active_sh_degree + 1) ** 2
        if bands <= 1:
            colors = p["sh0"]
        elif bands - 1 >= p["shN"].shape[1]:
            colors = torch.cat([p["sh0"], p["shN"]], dim=1)
        else:
            colors = torch.cat([p["sh0"], p["shN"][:, : bands - 1]], dim=1)
        rgb, alpha, info = rasterization(
            p["means"],
            p["quats"],
            p["scales"].exp(),
            p["opacities"].sigmoid(),
            colors.contiguous(),
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=self.active_sh_degree,
            packed=self.packed,
            absgrad=self.absgrad,
        )
        return rgb[0], alpha[0], info

    # --- optimization ------------------------------------------------------------
    def zero_grad(self):
        for o in self.opts.values():
            o.zero_grad(set_to_none=True)

    def step(self):
        for o in self.opts.values():
            o.step()

    def state_dict(self):
        return {
            "params": {k: v.detach().cpu() for k, v in self.params.items()},
            "extent": self.extent,
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
        }


def means_lr_at(step, total, extent, decay_to=0.01):
    """Exponential decay on the means LR, as in the 3DGS reference schedule."""
    base = LRS["means"] * extent
    if total <= 1:
        return base
    return base * math.exp(math.log(decay_to) * min(step / total, 1.0))
