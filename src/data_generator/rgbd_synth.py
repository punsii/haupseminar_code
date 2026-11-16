#!/usr/bin/env python3
"""
rgbd_synth.py - synthetic multi-view RGB-D sequence generator (Python stdlib only).

Scene: 8 x 8 x 3 m room with static props and a dynamic object floating in the
centre (rotating/bobbing cube + three orbiting, pulsating, spinning spheres).
All cameras are static and render the same timestamp t = frame / fps.

Output (per camera i):
    out/cam{i}/rgb/000000.png      8-bit RGB
    out/cam{i}/depth/000000.png    16-bit z-depth in mm (0 = no surface)
    out/cam{i}/mask/000000.png     8-bit, 255 where the dynamic object is visible
    out/cam{i}/static_*.png        background plate without the object (full mode only)
    out/cameras.json               intrinsics K, camera-to-world c2w, fps, conventions

Conventions: world z-up, metres, floor at z = 0. Cameras use OpenCV axes
(x right, y down, z forward). Pixel (u, v) is sampled at (u + 0.5, v + 0.5).

Usage:
    python rgbd_synth.py                                  # full scene, 60 frames @ 15 fps, 1080p
    python rgbd_synth.py --mode object                    # centre object only (black bg, depth 0)
    python rgbd_synth.py --width 480 --height 270 --frames 15   # quick preview
    python rgbd_synth.py --video                          # also encode videos (needs ffmpeg on PATH)

The static part is ray-cast once per camera; per frame only the pixels covered by
the dynamic object's projected bounding boxes are ray-cast and z-composited on top.
There are no shadows, so the cached background stays exact.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from array import array

# ----------------------------------------------------------------------------- config
# One entry per camera: (azimuth [deg], horizontal distance to object [m], height [m])
CAMERAS = [(0.0, 2.6, 1.6), (120.0, 2.6, 1.6), (240.0, 2.6, 1.6)]
HFOV_DEG = 70.0
OBJECT_CENTER = (0.0, 0.0, 1.0)  # dynamic object centre, all cameras look at it
LIGHT_POS = (0.6, -0.8, 2.8)
AMBIENT, DIFFUSE = 0.35, 1.0
DEPTH_SCALE = 1000.0  # depth png value per metre (-> mm)
BAND = 16  # image rows per task for the static render

INF = float("inf")
EPS = 1e-6


# ----------------------------------------------------------------------------- math
def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    s = 1.0 / math.sqrt(dot(a, a))
    return (a[0] * s, a[1] * s, a[2] * s)


def rot(axis, ang):
    """Rotation matrix (rows) for angle `ang` around `axis`, maps local -> world."""
    x, y, z = norm(axis)
    c, s = math.cos(ang), math.sin(ang)
    k = 1.0 - c
    return ((c + x * x * k, x * y * k - z * s, x * z * k + y * s),
            (y * x * k + z * s, c + y * y * k, y * z * k - x * s),
            (z * x * k - y * s, z * y * k + x * s, c + z * z * k))


def mat_vec(R, v):
    return (dot(R[0], v), dot(R[1], v), dot(R[2], v))


def mat_t_vec(R, v):
    return (R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2])


def hash01(i, j, k=0):
    """Deterministic pseudo-random value in [0, 1] for an integer cell."""
    h = (i * 73856093 ^ j * 19349663 ^ k * 83492791) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 0x2C1B3C6D) & 0xFFFFFFFF
    return ((h ^ (h >> 12)) & 0xFFFF) / 65535.0


# ----------------------------------------------------------------------------- textures
# signature: tex(p_world, q_local, n_local) -> albedo (r, g, b) in [0, 1]
WALL_COLOURS = ((0.80, 0.55, 0.50), (0.55, 0.70, 0.82), (0.60, 0.78, 0.55), (0.85, 0.80, 0.55))
CUBE_COLOURS = ((0.9, 0.2, 0.2), (0.2, 0.8, 0.8), (0.2, 0.8, 0.2),
                (0.8, 0.2, 0.8), (0.2, 0.3, 0.9), (0.9, 0.8, 0.2))
SPHERE_COLOURS = (((0.95, 0.35, 0.10), (0.95, 0.95, 0.90)),
                  ((0.10, 0.60, 0.95), (0.95, 0.95, 0.90)),
                  ((0.30, 0.85, 0.25), (0.15, 0.15, 0.20)))


def tex_room(p, q, n):
    x, y, z = p
    if n[2] > 0.5:  # floor: 0.5 m checker tiles with per-tile variation
        i, j = math.floor(x * 2.0), math.floor(y * 2.0)
        s = (0.75 if (i + j) & 1 else 0.5) * (0.85 + 0.15 * hash01(i, j))
        return (s, 0.85 * s, 0.65 * s)
    if n[2] < -0.5:  # ceiling
        return (0.85, 0.85, 0.82)
    if z < 0.12:  # skirting board
        return (0.25, 0.22, 0.20)
    w = (0 if n[0] > 0 else 1) if abs(n[0]) > 0.5 else (2 if n[1] > 0 else 3)
    u = y if abs(n[0]) > 0.5 else x
    s = 0.7 + 0.3 * hash01(math.floor(u * 4.0), math.floor(z * 4.0), w)  # 25 cm tiles
    c = WALL_COLOURS[w]
    return (c[0] * s, c[1] * s, c[2] * s)


def tex_checker(c1, c2, cell):
    def tex(p, q, n):  # 2D checker on each box face (constant axis skipped)
        k = sum(math.floor(qi / cell) for qi, ni in zip(q, n) if not ni)
        return c1 if k & 1 else c2
    return tex


def tex_cube(p, q, n):
    a = 0 if n[0] else (1 if n[1] else 2)
    c = CUBE_COLOURS[2 * a + (n[a] > 0)]
    k = sum(math.floor(qi * 8.0) for qi, ni in zip(q, n) if not ni)
    s = 1.0 if k & 1 else 0.6
    return (c[0] * s, c[1] * s, c[2] * s)


def tex_beachball(c1, c2):
    def tex(p, q, n):  # q = local unit direction
        lon = math.floor((math.atan2(q[1], q[0]) / math.pi + 1.0) * 4.0)  # 8 segments
        lat = math.floor((q[2] + 1.0) * 1.5)  # 3 bands
        return c1 if (lon + lat) & 1 else c2
    return tex


# ----------------------------------------------------------------------------- primitives
class Box:
    """Box with half extents `half`, rotation R (local->world rows, None = axis aligned).
    inside=True: the viewer is inside (room) -> far side is hit, normals point inward."""

    def __init__(self, center, half, tex, R=None, inside=False):
        self.c, self.h, self.tex, self.R, self.inside = center, half, tex, R, inside

    def bounds(self):
        if self.R is None:
            e = self.h
        else:
            e = tuple(sum(abs(self.R[i][j]) * self.h[j] for j in range(3)) for i in range(3))
        return sub(self.c, e), add(self.c, e)

    def hit(self, o, d):
        q = sub(o, self.c)
        if self.R is not None:
            q, d = mat_t_vec(self.R, q), mat_t_vec(self.R, d)
        tn, tf = -INF, INF
        for qi, di, hi in zip(q, d, self.h):
            if di == 0.0:
                if abs(qi) > hi:
                    return INF
                continue
            t1, t2 = (-hi - qi) / di, (hi - qi) / di
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tn:
                tn = t1
            if t2 < tf:
                tf = t2
        if tn > tf:
            return INF
        t = tf if self.inside else tn
        return t if t > EPS else INF

    def surface(self, p):
        q = sub(p, self.c)
        if self.R is not None:
            q = mat_t_vec(self.R, q)
        a = max((0, 1, 2), key=lambda i: abs(q[i]) / self.h[i])
        nl = [0.0, 0.0, 0.0]
        nl[a] = (1.0 if q[a] > 0.0 else -1.0) * (-1.0 if self.inside else 1.0)
        nl = tuple(nl)
        n = nl if self.R is None else mat_vec(self.R, nl)
        return n, self.tex(p, q, nl)


class Sphere:
    def __init__(self, center, radius, tex, R=None):
        self.c, self.r, self.tex, self.R = center, radius, tex, R

    def bounds(self):
        e = (self.r, self.r, self.r)
        return sub(self.c, e), add(self.c, e)

    def hit(self, o, d):
        ox, oy, oz = o[0] - self.c[0], o[1] - self.c[1], o[2] - self.c[2]
        a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        b = ox * d[0] + oy * d[1] + oz * d[2]
        disc = b * b - a * (ox * ox + oy * oy + oz * oz - self.r * self.r)
        if disc < 0.0:
            return INF
        s = math.sqrt(disc)
        t = (-b - s) / a
        if t > EPS:
            return t
        t = (-b + s) / a
        return t if t > EPS else INF

    def surface(self, p):
        n = ((p[0] - self.c[0]) / self.r, (p[1] - self.c[1]) / self.r, (p[2] - self.c[2]) / self.r)
        nl = n if self.R is None else mat_t_vec(self.R, n)
        return n, self.tex(p, nl, nl)


# ----------------------------------------------------------------------------- scene
def static_scene():
    """Room + props. Never changes over time."""
    wood = tex_checker((0.55, 0.38, 0.22), (0.42, 0.28, 0.15), 0.2)
    return [
        Box((0.0, 0.0, 1.5), (4.0, 4.0, 1.5), tex_room, inside=True),  # room 8 x 8 x 3 m
        Box((2.9, 2.9, 0.4), (0.4, 0.4, 0.4), wood, R=rot((0, 0, 1), math.radians(20))),  # crate
        Box((-3.5, 1.2, 0.9), (0.3, 0.8, 0.9),
            tex_checker((0.30, 0.35, 0.50), (0.50, 0.55, 0.70), 0.3)),  # cabinet
        Box((2.7, -2.6, 0.3), (0.6, 0.4, 0.3),
            tex_checker((0.70, 0.70, 0.70), (0.35, 0.35, 0.40), 0.15)),  # bench
        Sphere((-2.6, -2.9, 0.5), 0.5, tex_beachball((0.9, 0.75, 0.1), (0.2, 0.2, 0.25))),  # ball
    ]


def dynamic_scene(t):
    """The animated centre object at time t [s]: rigid motion plus non-rigid parts."""
    cx, cy, cz = OBJECT_CENTER
    tau = 2.0 * math.pi
    prims = [Box((cx, cy, cz + 0.1 * math.sin(tau * 0.5 * t)), (0.25, 0.25, 0.25), tex_cube,
                 R=rot((0.3, 0.2, 1.0), tau * 0.25 * t))]
    for k, (c1, c2) in enumerate(SPHERE_COLOURS):
        ph = tau * k / 3.0
        ang = tau * 0.4 * t + ph
        rad = 0.7 + 0.08 * math.sin(tau * 0.7 * t + ph)
        pos = (cx + rad * math.cos(ang), cy + rad * math.sin(ang),
               cz + 0.25 * math.sin(tau * 0.3 * t + 2.0 * ph))
        r = 0.12 + 0.03 * math.sin(tau * 1.1 * t + ph)  # pulsating radius
        prims.append(Sphere(pos, r, tex_beachball(c1, c2), R=rot((0, 0, 1), tau * 0.8 * t + ph)))
    return prims


# ----------------------------------------------------------------------------- cameras
def make_cameras(W, H):
    f = 0.5 * W / math.tan(math.radians(HFOV_DEG) / 2.0)
    cams = []
    for az, dist, height in CAMERAS:
        a = math.radians(az)
        C = (OBJECT_CENTER[0] + dist * math.cos(a), OBJECT_CENTER[1] + dist * math.sin(a), height)
        fwd = norm(sub(OBJECT_CENTER, C))
        right = norm(cross(fwd, (0.0, 0.0, 1.0)))
        down = cross(fwd, right)
        cams.append({"C": C, "right": right, "down": down, "fwd": fwd,
                     "fx": f, "fy": f, "cx": W / 2.0, "cy": H / 2.0})
    return cams


def screen_bbox(cam, lo, hi, W, H):
    """Pixel range (u0, u1, v0, v1) an AABB can cover, None if not visible."""
    C, r, dn, f = cam["C"], cam["right"], cam["down"], cam["fwd"]
    us, vs, behind = [], [], 0
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                q = (x - C[0], y - C[1], z - C[2])
                zc = dot(q, f)
                if zc <= EPS:
                    behind += 1
                    continue
                us.append(cam["fx"] * dot(q, r) / zc + cam["cx"])
                vs.append(cam["fy"] * dot(q, dn) / zc + cam["cy"])
    if behind == 8:
        return None
    if behind:
        return (0, W, 0, H)  # straddles the camera plane: no culling
    u0, u1 = max(0, int(min(us)) - 1), min(W, int(max(us)) + 2)
    v0, v1 = max(0, int(min(vs)) - 1), min(H, int(max(vs)) + 2)
    return (u0, u1, v0, v1) if u0 < u1 and v0 < v1 else None


def cull(prims, cam, W, H):
    out = []
    for pr in prims:
        box = screen_bbox(cam, *pr.bounds(), W, H)
        if box:
            out.append((pr,) + box)
    return out


# ----------------------------------------------------------------------------- rendering
def row_hits(cam, culled, v):
    """Ray-cast image row v -> [(u, t, prim, ray)]; t is z-depth in metres because the
    ray's camera-z component is 1."""
    active = [(pr, u0, u1) for pr, u0, u1, v0, v1 in culled if v0 <= v < v1]
    if not active:
        return []
    C, r, f, dn = cam["C"], cam["right"], cam["fwd"], cam["down"]
    yc = (v + 0.5 - cam["cy"]) / cam["fy"]
    bx, by, bz = f[0] + dn[0] * yc, f[1] + dn[1] * yc, f[2] + dn[2] * yc
    fx, cx = cam["fx"], cam["cx"]
    hits = []
    for u in range(min(a[1] for a in active), max(a[2] for a in active)):
        xc = (u + 0.5 - cx) / fx
        ray = (bx + r[0] * xc, by + r[1] * xc, bz + r[2] * xc)
        tb, pb = INF, None
        for pr, u0, u1 in active:
            if u0 <= u < u1:
                t = pr.hit(C, ray)
                if t < tb:
                    tb, pb = t, pr
        if pb is not None:
            hits.append((u, tb, pb, ray))
    return hits


def shade(prim, C, ray, t):
    """Lambert + ambient, single point light, no shadows -> (r, g, b) bytes."""
    p = (C[0] + ray[0] * t, C[1] + ray[1] * t, C[2] + ray[2] * t)
    n, (r, g, b) = prim.surface(p)
    lx, ly, lz = LIGHT_POS[0] - p[0], LIGHT_POS[1] - p[1], LIGHT_POS[2] - p[2]
    d2 = lx * lx + ly * ly + lz * lz
    ndl = (n[0] * lx + n[1] * ly + n[2] * lz) / math.sqrt(d2)
    k = 255.0 * (AMBIENT + (DIFFUSE * ndl / (1.0 + 0.08 * d2) if ndl > 0.0 else 0.0))
    return (min(255, int(r * k)), min(255, int(g * k)), min(255, int(b * k)))


def to_mm(t):
    return min(65535, int(t * DEPTH_SCALE + 0.5))


G = {}  # per-worker state, set by init_worker


def init_worker(cfg, static):
    G.update(cfg)
    G["static"] = static


def render_static_band(task):
    ci, v0, v1 = task
    W, cam = G["W"], G["cams"][ci]
    culled = cull(static_scene(), cam, W, G["H"])
    rgb = bytearray(3 * W * (v1 - v0))
    dep = array("H", bytes(2 * W * (v1 - v0)))
    for v in range(v0, v1):
        row = (v - v0) * W
        for u, t, pr, ray in row_hits(cam, culled, v):
            i = row + u
            rgb[3 * i:3 * i + 3] = shade(pr, cam["C"], ray, t)
            dep[i] = to_mm(t)
    return ci, v0, bytes(rgb), dep.tobytes()


def render_frame(task):
    f, ci = task
    W, H, cam = G["W"], G["H"], G["cams"][ci]
    if G["static"]:  # full mode: start from the cached background
        rgb = bytearray(G["static"][ci][0])
        dep = array("H", G["static"][ci][1])
    else:  # object mode: black background, invalid depth
        rgb = bytearray(3 * W * H)
        dep = array("H", bytes(2 * W * H))
    mask = bytearray(W * H)
    culled = cull(dynamic_scene(f / G["fps"]), cam, W, H)
    if culled:
        for v in range(min(c[3] for c in culled), max(c[4] for c in culled)):
            for u, t, pr, ray in row_hits(cam, culled, v):
                i = v * W + u
                mm = to_mm(t)
                if dep[i] and dep[i] <= mm:  # static geometry in front
                    continue
                rgb[3 * i:3 * i + 3] = shade(pr, cam["C"], ray, t)
                dep[i] = mm
                mask[i] = 255
    d, name = os.path.join(G["out"], f"cam{ci}"), f"{f:06d}.png"
    write_png(os.path.join(d, "rgb", name), rgb, W, H, 3)
    write_depth_png(os.path.join(d, "depth", name), dep, W, H)
    write_png(os.path.join(d, "mask", name), mask, W, H, 1)
    return f, ci


# ----------------------------------------------------------------------------- output
def write_png(path, data, w, h, channels, bit_depth=8):
    stride = w * channels * bit_depth // 8
    mv = memoryview(data)
    raw = b"".join(b"\x00" + mv[y * stride:(y + 1) * stride] for y in range(h))

    def chunk(tag, body):
        crc = zlib.crc32(body, zlib.crc32(tag))
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    colour_type = {1: 0, 3: 2}[channels]
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, colour_type, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6))
                 + chunk(b"IEND", b""))


def write_depth_png(path, dep, w, h):
    if sys.byteorder == "little":  # PNG stores 16-bit samples big-endian
        dep = array("H", dep.tobytes())
        dep.byteswap()
    write_png(path, dep.tobytes(), w, h, 1, 16)


def write_meta(out, cams, args):
    W, H = args.width, args.height
    meta = {
        "width": W, "height": H, "fps": args.fps, "num_frames": args.frames, "mode": args.mode,
        "depth_scale": DEPTH_SCALE,
        "depth": "uint16 png, z-depth along the optical axis, metres = value / depth_scale, 0 = invalid",
        "world": "right-handed, z-up, metres, floor at z=0",
        "camera_convention": "OpenCV (x right, y down, z forward); c2w = camera-to-world",
        "pixel_center_offset": 0.5,
        "object_center": OBJECT_CENTER,
        "cameras": [],
    }
    for i, c in enumerate(cams):
        r, d, f, C = c["right"], c["down"], c["fwd"], c["C"]
        meta["cameras"].append({
            "name": f"cam{i}", "fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"],
            "K": [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
            "c2w": [[r[0], d[0], f[0], C[0]], [r[1], d[1], f[1], C[1]],
                    [r[2], d[2], f[2], C[2]], [0.0, 0.0, 0.0, 1.0]],
        })
    with open(os.path.join(out, "cameras.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def encode_videos(out, n_cams, fps):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found on PATH, skipping videos", file=sys.stderr)
        return
    jobs = (("rgb", "mp4", ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]),
            ("depth", "mkv", ["-c:v", "ffv1"]),  # lossless, keeps 16 bit
            ("mask", "mkv", ["-c:v", "ffv1"]))
    for ci in range(n_cams):
        for kind, ext, codec in jobs:
            src = os.path.join(out, f"cam{ci}", kind, "%06d.png")
            dst = os.path.join(out, f"cam{ci}_{kind}.{ext}")
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src,
                            *codec, dst], check=True)


def progress(label, k, n, t0):
    el = time.time() - t0
    print(f"\r{label}: {k}/{n}  {el:.0f}s elapsed, ~{el / k * (n - k):.0f}s left   ",
          end="\n" if k == n else "", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("full", "object"), default="full",
                    help="full scene, or only the dynamic centre object")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default="rgbd_out")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--video", action="store_true", help="also encode videos with ffmpeg")
    args = ap.parse_args()

    W, H = args.width, args.height
    cams = make_cameras(W, H)
    for ci in range(len(cams)):
        for kind in ("rgb", "depth", "mask"):
            os.makedirs(os.path.join(args.out, f"cam{ci}", kind), exist_ok=True)
    write_meta(args.out, cams, args)
    cfg = {"W": W, "H": H, "fps": args.fps, "out": args.out, "cams": cams}
    t0 = time.time()

    static = None
    if args.mode == "full":
        rgbs = [bytearray(3 * W * H) for _ in cams]
        deps = [bytearray(2 * W * H) for _ in cams]
        tasks = [(ci, v, min(v + BAND, H)) for ci in range(len(cams)) for v in range(0, H, BAND)]
        with mp.Pool(args.workers, init_worker, (cfg, None)) as pool:
            for k, (ci, v0, rgb, dep) in enumerate(pool.imap_unordered(render_static_band, tasks), 1):
                rgbs[ci][3 * W * v0:3 * W * v0 + len(rgb)] = rgb
                deps[ci][2 * W * v0:2 * W * v0 + len(dep)] = dep
                progress("static background", k, len(tasks), t0)
        static = [(bytes(r), bytes(d)) for r, d in zip(rgbs, deps)]
        for ci, (r, d) in enumerate(static):
            write_png(os.path.join(args.out, f"cam{ci}", "static_rgb.png"), r, W, H, 3)
            write_depth_png(os.path.join(args.out, f"cam{ci}", "static_depth.png"), array("H", d), W, H)

    t1 = time.time()
    tasks = [(f, ci) for f in range(args.frames) for ci in range(len(cams))]
    with mp.Pool(args.workers, init_worker, (cfg, static)) as pool:
        for k, _ in enumerate(pool.imap_unordered(render_frame, tasks), 1):
            progress("frames", k, len(tasks), t1)

    if args.video:
        encode_videos(args.out, len(cams), args.fps)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
