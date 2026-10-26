"""[Learn, Render, Learn] on a static scene.

Frames arrive in video order. For each frame: render it from its camera *before* training
on it (= prediction of a view the model hasn't trained on), then take a few optimization
steps on it plus random earlier frames. Poses and init points come from COLMAP.
"""
import argparse
import os
import time

import imageio.v3 as iio
import torch
import torch.nn.functional as F

from render import _quat_to_rotmat, render

p = argparse.ArgumentParser()
p.add_argument("--data", default="../data/scene/undistorted", help="output dir of colmap image_undistorter")
p.add_argument("--scale", type=float, default=1.0, help="image downscale, e.g. 0.2 on CPU")
p.add_argument("--steps", type=int, default=30, help="optimization steps per incoming frame")
p.add_argument("--out", default="out")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def read_txt(path):
    return [line for line in open(path).read().splitlines() if not line.startswith("#")]


# --- COLMAP text model (PINHOLE after undistortion) ---
sparse = f"{args.data}/sparse"
cam = read_txt(f"{sparse}/cameras.txt")[0].split()
assert cam[1] == "PINHOLE", f"expected an undistorted PINHOLE camera, got {cam[1]}"
W0, H0 = int(cam[2]), int(cam[3])
fx, fy, cx, cy = map(float, cam[4:8])
W, H = round(W0 * args.scale), round(H0 * args.scale)
sx, sy = W / W0, H / H0
K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], device=dev)

frames = []  # (image name, world-to-camera 4x4)
for line in read_txt(f"{sparse}/images.txt")[0::2]:  # 2 lines per image; the 2nd lists 2D points
    e = line.split()
    viewmat = torch.eye(4)
    viewmat[:3, :3] = _quat_to_rotmat(torch.tensor([float(v) for v in e[1:5]])[None])[0]
    viewmat[:3, 3] = torch.tensor([float(v) for v in e[5:8]])
    frames.append((e[9], viewmat.to(dev)))
frames.sort(key=lambda f: f[0])  # video order (ffmpeg's %05d names)


def load(name):
    img = torch.from_numpy(iio.imread(f"{args.data}/images/{name}")[..., :3]).to(dev)
    img = img.permute(2, 0, 1)[None].float() / 255
    return F.interpolate(img, size=(H, W), mode="area")[0].permute(1, 2, 0)


gts = [load(name) for name, _ in frames]
pts = torch.tensor([[float(v) for v in line.split()[1:7]] for line in read_txt(f"{sparse}/points3D.txt")])
xyz, rgb = pts[:, :3], pts[:, 3:] / 255

# --- Gaussians initialized from the sparse points (no densification yet) ---
nn_dist = torch.cat([torch.cdist(c, xyz).topk(4, largest=False).values[:, 1:].mean(1) for c in xyz.split(4096)])
params = {
    "means": xyz,
    "scales": nn_dist.clamp_min(1e-6).log()[:, None].repeat(1, 3),  # log-space, start isotropic
    "quats": torch.tensor([1.0, 0, 0, 0]).repeat(len(xyz), 1),
    "opacities": torch.full((len(xyz),), 0.1).logit(),
    "colors": rgb.clamp(0.01, 0.99).logit(),
}
params = {k: v.to(dev).requires_grad_() for k, v in params.items()}
centers = torch.stack([-v[:3, :3].T @ v[:3, 3] for _, v in frames])
extent = 1.1 * (centers - centers.mean(0)).norm(dim=1).max().item()
lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}  # 3DGS defaults
opt = torch.optim.Adam([{"params": [params[k]], "lr": lr} for k, lr in lrs.items()], eps=1e-15)


def render_view(viewmat):
    g = params
    img, _ = render(g["means"], g["quats"], g["scales"].exp(), g["opacities"].sigmoid(), g["colors"].sigmoid(), viewmat, K, W, H)
    return img


print(f"{dev} | {len(frames)} frames at {W}x{H} | {len(xyz)} Gaussians | {args.steps} steps/frame")
os.makedirs(args.out, exist_ok=True)
t0 = time.perf_counter()
for t, (name, viewmat) in enumerate(frames):
    # Render: predict the incoming frame before learning on it
    with torch.no_grad():
        pred = render_view(viewmat).clamp(0, 1)
    psnr = -10 * torch.log10(F.mse_loss(pred, gts[t])).item()
    iio.imwrite(f"{args.out}/{t:05d}.png", (torch.cat([gts[t], pred], 1) * 255).byte().cpu().numpy())

    # Learn: newest frame half the time, otherwise a random earlier one (so old views aren't forgotten)
    for _ in range(args.steps):
        i = t if torch.rand(()) < 0.5 else int(torch.randint(0, t + 1, ()))
        loss = (render_view(frames[i][1]) - gts[i]).abs().mean()  # L1
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"frame {t + 1:3d}/{len(frames)} {name}: predicted PSNR {psnr:5.2f} dB | {time.perf_counter() - t0:6.1f}s")

torch.save({k: v.detach().cpu() for k, v in params.items()}, f"{args.out}/gaussians.pt")
