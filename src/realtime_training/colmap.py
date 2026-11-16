"""Reading COLMAP's output: where each photo was taken from, and a rough point cloud.

COLMAP works out, from the photos alone, where the camera was for each one and roughly
what the scene geometry is. Training needs both: the camera positions to know where to
render from, and the points as a starting guess for where to put Gaussians.

Expects the output of `colmap image_undistorter` followed by
`colmap model_converter --output_type TXT` (both in colmap.sh), that is a directory with

    <data>/images/*.jpg
    <data>/sparse/{cameras,images,points3D}.txt

The text form is required; the .bin form COLMAP writes by default is not read here.

A note on conventions, since getting these wrong produces a model that trains to garbage
without any error: a "viewmat" here converts world coordinates into camera coordinates
(not the other way round), and the camera looks along +z with +y pointing down. This
matches what gsplat expects and what src/data_generator/rgbd_synth.py writes.
"""
import os
from dataclasses import dataclass

import torch


def read_txt(path):
    return [line for line in open(path).read().splitlines() if not line.startswith("#")]


def quat_to_rotmat(q):
    """COLMAP/gsplat quaternion order (w, x, y, z) -> rotation matrices [..., 3, 3]."""
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(-1, 3, 3)


@dataclass
class Scene:
    image_dir: str
    names: list  # image file names, sorted (= capture order for %05d frame dumps)
    viewmats: torch.Tensor  # [N, 4, 4] world-to-camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int  # native image size, before any --scale
    height: int
    points: torch.Tensor  # [P, 3] sparse point cloud
    point_colors: torch.Tensor  # [P, 3] in [0, 1]
    extent: float  # camera-spread radius; gsplat's scene_scale

    def __len__(self):
        return len(self.names)


def load_scene(data):
    """Read a COLMAP model. Returns everything on CPU; callers move what they need."""
    sparse = os.path.join(data, "sparse")
    for f in ("cameras.txt", "images.txt", "points3D.txt"):
        if not os.path.exists(os.path.join(sparse, f)):
            raise FileNotFoundError(
                f"{os.path.join(sparse, f)} missing. Run colmap.sh's model_converter step "
                "(--output_type TXT); the .bin model alone is not readable here."
            )

    cam = read_txt(os.path.join(sparse, "cameras.txt"))[0].split()
    if cam[1] != "PINHOLE":
        raise ValueError(f"expected an undistorted PINHOLE camera, got {cam[1]}")
    width, height = int(cam[2]), int(cam[3])
    fx, fy, cx, cy = map(float, cam[4:8])

    # images.txt stores 2 lines per image; the second lists 2D keypoints, so stride by 2.
    frames = []
    for line in read_txt(os.path.join(sparse, "images.txt"))[0::2]:
        e = line.split()
        viewmat = torch.eye(4)
        viewmat[:3, :3] = quat_to_rotmat(torch.tensor([float(v) for v in e[1:5]])[None])[0]
        viewmat[:3, 3] = torch.tensor([float(v) for v in e[5:8]])
        frames.append((e[9], viewmat))
    if not frames:
        raise ValueError(f"no registered images in {sparse}/images.txt")
    frames.sort(key=lambda f: f[0])
    names = [n for n, _ in frames]
    viewmats = torch.stack([v for _, v in frames])

    pts = torch.tensor([[float(v) for v in line.split()[1:7]]
                        for line in read_txt(os.path.join(sparse, "points3D.txt"))])

    # Camera centers: C = -R^T t. Extent drives the means learning rate and densification.
    centers = torch.stack([-v[:3, :3].T @ v[:3, 3] for v in viewmats])
    extent = 1.1 * (centers - centers.mean(0)).norm(dim=1).max().item()

    return Scene(
        image_dir=os.path.join(data, "images"),
        names=names, viewmats=viewmats,
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
        points=pts[:, :3], point_colors=pts[:, 3:] / 255,
        extent=extent,
    )


def holdout_split(n, every=8):
    """Set aside every `every`-th view so it is never trained on.

    Scoring the model on views it never saw is the only honest quality measure, and it is
    what published 3D Gaussian Splatting numbers use. `every=0` disables it.
    """
    test = list(range(0, n, every)) if every > 0 else []
    train = [i for i in range(n) if i not in set(test)]
    return train, test


def scene_name(data):
    """Scene name taken from a data path, used to name output directories.

        ../data/scene2/undistorted  ->  "scene2"
        ../data/scene2              ->  "scene2"
    """
    p = os.path.normpath(os.path.abspath(data))
    # Walk up past the directory names COLMAP creates, so the scene folder itself is named.
    while os.path.basename(p) in ("undistorted", "sparse", "images", "0"):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.basename(p) or "scene"
