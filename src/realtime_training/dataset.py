"""Loading photos on demand, keeping only a few in memory.

loop.py decoded every photo up front and held them all at once. On full-size 12-megapixel
stills that is about 144 MB per photo, so a couple of hundred photos exhausted 30 GB of
memory plus 45 GB of swap before being killed. Here photos are decoded when asked for and
only `cache_size` of them are kept, least-recently-used first out.

This class also handles `--scale`. Shrinking the photos shrinks the camera's focal length
and image centre by the same factor, which has to be done together or the renders will
not line up with the photos.
"""
import os
from collections import OrderedDict

import imageio.v3 as iio
import torch
import torch.nn.functional as F


class Images:
    def __init__(self, scene, scale=1.0, device="cuda", cache_size=64):
        self.scene = scene
        self.device = device
        self.cache_size = max(1, cache_size)
        self._cache = OrderedDict()

        self.width = round(scene.width * scale)
        self.height = round(scene.height * scale)
        # Intrinsics scale with the image; cx/cy follow because pixel centers are at +0.5.
        sx, sy = self.width / scene.width, self.height / scene.height
        self.K = torch.tensor([
            [scene.fx * sx, 0.0, scene.cx * sx],
            [0.0, scene.fy * sy, scene.cy * sy],
            [0.0, 0.0, 1.0],
        ], device=device)
        self.viewmats = scene.viewmats.to(device)

    def __len__(self):
        return len(self.scene)

    def __getitem__(self, i):
        """Ground-truth image i as [H, W, 3] float in [0, 1], on `device`."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]

        path = os.path.join(self.scene.image_dir, self.scene.names[i])
        img = torch.from_numpy(iio.imread(path)[..., :3].copy()).to(self.device)
        img = img.permute(2, 0, 1)[None].float() / 255
        if (img.shape[-2], img.shape[-1]) != (self.height, self.width):
            img = F.interpolate(img, size=(self.height, self.width), mode="area")
        img = img[0].permute(1, 2, 0).contiguous()

        self._cache[i] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def viewmat(self, i):
        return self.viewmats[i]
