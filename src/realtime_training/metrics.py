"""Two ways of scoring a rendered image against the photo it should match.

PSNR (peak signal-to-noise ratio) is average pixel error expressed in decibels, where
higher is better. It is the number everyone quotes, so it is what comparisons use, but it
is a crude measure of how an image actually looks. As a rough feel, on colours scaled to
0..1: 9 dB is noise, 20 dB is recognizable but blurry, 27-31 dB is published Gaussian
Splatting quality.

SSIM (structural similarity) instead compares local patches - brightness, contrast and
pattern - and returns 1.0 for identical images. It tracks perceived sharpness much better
than PSNR, which is why the training loss blends the two: pixel error alone is happy with
a blurry answer.

SSIM is written out here rather than taken from a library to avoid another dependency.
The patch weighting follows the usual choice, an 11-pixel Gaussian window.
"""
import torch
import torch.nn.functional as F

_C1, _C2 = 0.01**2, 0.03**2


def psnr(pred, gt):
    """Both [H, W, 3] in [0, 1]."""
    return (-10 * torch.log10(F.mse_loss(pred, gt))).item()


def _gaussian_window(size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]  # [1, 1, size, size]


_window_cache = {}


def ssim(pred, gt, size=11, sigma=1.5):
    """Mean SSIM over channels. Both [H, W, 3] in [0, 1]. Differentiable."""
    x = pred.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    c = x.shape[1]

    key = (size, sigma, x.device, x.dtype, c)
    if key not in _window_cache:
        _window_cache[key] = _gaussian_window(size, sigma, x.device, x.dtype).expand(c, 1, size, size).contiguous()
    w = _window_cache[key]

    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy

    num = (2 * mu_xy + _C1) * (2 * sigma_xy + _C2)
    den = (mu_x2 + mu_y2 + _C1) * (sigma_x2 + sigma_y2 + _C2)
    return (num / den).mean()


def photometric_loss(pred, gt, lambda_dssim=0.2):
    """The quantity training minimizes: average pixel error blended with 1 - SSIM.

    `lambda_dssim` is how much weight the structural term gets; 0 leaves plain pixel
    error. Dropping it is worth roughly 0.5-1 dB of final PSNR.
    """
    l1 = (pred - gt).abs().mean()
    if lambda_dssim <= 0:
        return l1, l1, None
    s = ssim(pred, gt)
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - s), l1, s
