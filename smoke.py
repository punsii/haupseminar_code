import time

import imageio.v3 as iio
import torch

from render import render

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", dev)

torch.manual_seed(0)
N, W, H = 2000, 640, 480
means = (torch.randn(N, 3) * 0.5).to(dev).requires_grad_()
quats = torch.nn.functional.normalize(torch.randn(N, 4), dim=-1).to(dev)
scales = (torch.rand(N, 3) * 0.05).to(dev)
opacities = torch.rand(N).to(dev)
colors = torch.rand(N, 3).to(dev)

viewmat = torch.eye(4, device=dev)
viewmat[2, 3] = 3.0  # world-to-camera: scene sits 3 units in front of the camera
K = torch.tensor([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]], device=dev)

t0 = time.perf_counter()
img, alpha = render(means, quats, scales, opacities, colors, viewmat, K, W, H)  # CUDA: first call JIT-compiles gsplat
img.mean().backward()  # the loop will need gradients, so check they flow
print(f"render + backward: {time.perf_counter() - t0:.2f}s | grad norm: {means.grad.norm():.4f}")

iio.imwrite(f"smoke_{dev}.png", (img.detach().clamp(0, 1) * 255).byte().cpu().numpy())
print(f"wrote smoke_{dev}.png")
