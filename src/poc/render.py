"""render(): gsplat on CUDA, naive differentiable PyTorch rasterizer everywhere else.

Conventions follow gsplat: quats wxyz, scales/opacities already activated,
viewmat is world-to-camera (OpenCV: +z forward), pixel centers at +0.5, black background.
"""
import torch


def render(means, quats, scales, opacities, colors, viewmat, K, W, H):
    """Returns rgb [H,W,3] and alpha [H,W,1]."""
    if means.is_cuda:
        from gsplat import rasterization

        rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, viewmat[None], K[None], W, H)
        return rgb[0], alpha[0]
    return _render_torch(means, quats, scales, opacities, colors, viewmat, K, W, H)


def _quat_to_rotmat(q):
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(-1, 3, 3)


def _render_torch(means, quats, scales, opacities, colors, viewmat, K, W, H, tile=32):
    # 1. world -> camera, drop Gaussians behind the near plane
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p = means @ R.T + t
    keep = p[:, 2] > 0.01
    p, quats, scales, opacities, colors = p[keep], quats[keep], scales[keep], opacities[keep], colors[keep]
    x, y, z = p.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mean2d = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)

    # 2. 3D covariance R S S^T R^T -> 2D covariance via the projection Jacobian (EWA)
    M = _quat_to_rotmat(quats) * scales[:, None, :]
    cov3d = M @ M.transpose(1, 2)
    zero = torch.zeros_like(z)
    J = torch.stack([fx / z, zero, -fx * x / z**2, zero, fy / z, -fy * y / z**2], -1).view(-1, 2, 3)
    T = J @ R
    cov2d = T @ cov3d @ T.transpose(1, 2) + 0.3 * torch.eye(2, device=p.device, dtype=p.dtype)  # gsplat's eps2d
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)  # inverse covariance (A, B, C)
    radius = 3 * torch.sqrt(0.5 * (a + c) + torch.sqrt((0.5 * (a - c)) ** 2 + b * b)).detach()

    # 3. front-to-back order
    order = torch.argsort(z)
    mean2d, conic, opacities, colors, radius = (v[order] for v in (mean2d, conic, opacities, colors, radius))
    m = mean2d.detach()

    # 4. alpha-composite per tile, only with Gaussians whose 3-sigma box touches the tile
    rgb = torch.zeros(H, W, 3, device=p.device, dtype=p.dtype)
    acc = torch.zeros(H, W, 1, device=p.device, dtype=p.dtype)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            hit = (m[:, 0] + radius > x0) & (m[:, 0] - radius < x1) & (m[:, 1] + radius > y0) & (m[:, 1] - radius < y1)
            if not hit.any():
                continue
            ys, xs = torch.meshgrid(
                torch.arange(y0, y1, device=p.device) + 0.5, torch.arange(x0, x1, device=p.device) + 0.5, indexing="ij"
            )
            d = torch.stack([xs, ys], -1).view(1, -1, 2) - mean2d[hit][:, None]  # [n, P, 2]
            A, B, C = conic[hit].unbind(-1)
            power = -0.5 * (A[:, None] * d[..., 0] ** 2 + C[:, None] * d[..., 1] ** 2) - B[:, None] * d[..., 0] * d[..., 1]
            alpha = (opacities[hit][:, None] * torch.exp(power)).clamp(max=0.99)
            alpha = alpha * (alpha >= 1 / 255)  # skip negligible contributions, like gsplat
            trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha[:-1]]), 0)
            w = trans * alpha  # [n, P]
            rgb[y0:y1, x0:x1] = (w.T @ colors[hit]).view(y1 - y0, x1 - x0, 3)
            acc[y0:y1, x0:x1] = w.sum(0).view(y1 - y0, x1 - x0, 1)
    return rgb, acc
