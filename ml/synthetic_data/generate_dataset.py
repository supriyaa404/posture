#!/usr/bin/env python3
"""
POSTURE - Phase 2: synthetic exercise image dataset generator.

Generates stylised (cartoon-like) human figures performing 7 physiotherapy
exercises, with correct and deliberately incorrect form, from varied camera
viewpoints, together with consistent 13-joint annotations.

IMPORTANT LIMITS (also written into dataset_summary.txt):
  * These are rendered stick/capsule figures, NOT real people or real webcam
    footage. Joint annotations are exact by construction; they are not
    MediaPipe outputs.
  * Form labels come from the generator's parameters, not from clinical
    assessment. At some camera angles an "incorrect" sample can look the same
    as a "correct" one (e.g. hip abduction seen from the side).
  * Training or testing on this data does NOT establish clinical accuracy.

Usage (from the repository root, with the virtual environment active):
    python ml/synthetic_data/generate_dataset.py --seed 42
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SCHEMA_VERSION = "1.0"
D2R = math.pi / 180.0

# 13 joints used by PostureNet (same order everywhere).
JOINT_NAMES = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
# Index of each joint in MediaPipe Pose's 33 landmarks (for later phases).
MEDIAPIPE_INDEX = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# exercise -> list of error types we can generate
EXERCISES: dict[str, list[str]] = {
    "shoulder_flexion": ["limited_range", "elbow_bent", "trunk_lean_back", "shoulder_shrug"],
    "shoulder_abduction": ["limited_range", "elbow_bent", "trunk_lean_side", "shoulder_shrug", "arm_forward_drift"],
    "elbow_flexion": ["limited_range", "upper_arm_drift", "trunk_swing"],
    "hip_abduction": ["limited_range", "trunk_lean_side", "leg_forward_drift"],
    "squat": ["shallow_depth", "knee_valgus", "excessive_forward_lean", "lateral_weight_shift"],
    "marching": ["low_knee_lift", "trunk_lean_back", "torso_sway"],
    "seated_knee_extension": ["incomplete_extension", "thigh_lift", "trunk_lean_back"],
}

ERROR_DESCRIPTIONS = {
    "limited_range": "movement stops well short of the target range",
    "elbow_bent": "elbow bends while the arm should stay straight",
    "trunk_lean_back": "torso leans backward to compensate",
    "trunk_lean_side": "torso leans sideways to compensate",
    "shoulder_shrug": "shoulder is hiked up toward the ear",
    "arm_forward_drift": "arm drifts forward instead of moving sideways",
    "upper_arm_drift": "upper arm swings forward during the curl",
    "trunk_swing": "torso swings backward to lift the forearm",
    "leg_forward_drift": "leg drifts forward instead of moving sideways",
    "shallow_depth": "squat depth is too shallow",
    "knee_valgus": "knees collapse inward",
    "excessive_forward_lean": "torso leans too far forward",
    "lateral_weight_shift": "hips shift sideways over one foot",
    "low_knee_lift": "knee is not lifted high enough",
    "torso_sway": "torso sways sideways while marching",
    "incomplete_extension": "knee does not straighten far enough",
    "thigh_lift": "thigh lifts off the seat during extension",
}


# --------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def v3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def sag(a: float) -> np.ndarray:
    """Direction in the sagittal plane. a = angle from straight down, forward positive."""
    return v3(0.0, -math.cos(a), math.sin(a))


def front(a: float, side_sign: float) -> np.ndarray:
    """Direction in the frontal plane. a = angle from straight down, outward positive."""
    return v3(side_sign * math.sin(a), -math.cos(a), 0.0)


def hang(side_sign: float) -> np.ndarray:
    return front(5 * D2R, side_sign)


# --------------------------------------------------------------------------
# subjects (body + appearance), fixed per subject
# --------------------------------------------------------------------------
@dataclass
class Body:
    shoulder_w: float
    hip_w: float
    torso: float
    upper_arm: float
    forearm: float
    thigh: float
    shank: float
    build: float
    head_r: float


SKIN = [(255, 224, 189), (241, 194, 150), (224, 172, 105), (198, 134, 66), (141, 85, 36), (96, 60, 30)]
HAIR = [(20, 15, 12), (60, 40, 25), (110, 75, 40), (170, 130, 60), (150, 150, 150), (200, 90, 40)]
CLOTH = [(200, 40, 40), (40, 90, 200), (30, 150, 90), (240, 190, 40), (120, 60, 160),
         (240, 120, 40), (30, 30, 30), (235, 235, 235), (90, 90, 100), (0, 150, 160),
         (220, 90, 140), (110, 140, 60), (150, 80, 50), (20, 40, 90)]


def jitter_color(rng: np.random.Generator, c: tuple[int, int, int], amt: int = 18) -> tuple[int, int, int]:
    return tuple(int(np.clip(v + rng.integers(-amt, amt + 1), 0, 255)) for v in c)  # type: ignore


def pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def make_subject(idx: int, rng: np.random.Generator) -> dict:
    scale = rng.uniform(0.88, 1.08)
    body = Body(
        shoulder_w=rng.uniform(0.34, 0.46) * scale,
        hip_w=rng.uniform(0.17, 0.23) * scale,
        torso=rng.uniform(0.48, 0.58) * scale,
        upper_arm=rng.uniform(0.27, 0.33) * scale,
        forearm=rng.uniform(0.23, 0.28) * scale,
        thigh=rng.uniform(0.38, 0.46) * scale,
        shank=rng.uniform(0.37, 0.45) * scale,
        build=rng.uniform(0.85, 1.2),
        head_r=rng.uniform(0.09, 0.105) * scale,
    )
    app = {
        "skin": jitter_color(rng, pick(rng, SKIN), 8),
        "hair": jitter_color(rng, pick(rng, HAIR), 10),
        "shirt": jitter_color(rng, pick(rng, CLOTH)),
        "pants": jitter_color(rng, pick(rng, CLOTH)),
        "shoes": jitter_color(rng, pick(rng, CLOTH[6:9] + CLOTH[:2])),
        "sleeve": pick(rng, ["short", "long"]),
        "legwear": pick(rng, ["long", "long", "shorts"]),
    }
    return {"subject_id": f"subj{idx:03d}", "index": idx, "body": body, "appearance": app}


# --------------------------------------------------------------------------
# 3D pose construction
# --------------------------------------------------------------------------
def build_pose(ex: str, B: Body, p: float, err: str, sev: float,
               side: str, rng: np.random.Generator):
    """
    Return (points3d, meta).

    World frame (metres): +x = subject's LEFT, +y = up, +z = forward (the
    direction the subject faces). Ground is y = 0.
    p   = progress through the movement in [0, 1] (1 = peak of the repetition).
    err = "none" or one of EXERCISES[ex]; sev in [0, 1] sets error magnitude.
    """
    sgn = 1.0 if side == "left" else -1.0
    other = "right" if side == "left" else "left"
    lx = B.hip_w / 2
    lean_f = 0.0  # torso forward lean (rad), negative = backward
    lean_s = 0.0  # torso lean toward +x (rad)
    shrug = {"left": 0.0, "right": 0.0}
    arms = {"left": (hang(1.0), hang(1.0)), "right": (hang(-1.0), hang(-1.0))}
    chair = None
    hy = B.thigh + B.shank

    def standing():
        hc = v3(0, hy, 0)
        legs = {}
        for sd, s in (("left", 1.0), ("right", -1.0)):
            legs[sd] = (v3(s * lx, hy, 0), v3(s * lx, B.shank, 0), v3(s * lx, 0, 0))
        return hc, legs

    primary = 0.0
    peak_used = 0.0

    if ex == "shoulder_flexion":
        peak = lerp(100, 55, sev) if err == "limited_range" else rng.uniform(140, 170)
        a = p * peak * D2R
        bend = lerp(35, 85, sev) * D2R if err == "elbow_bent" else 0.0
        if err == "trunk_lean_back":
            lean_f = -lerp(8, 25, sev) * D2R * p
        if err == "shoulder_shrug":
            shrug[side] = lerp(0.025, 0.06, sev)
        arms[side] = (sag(a), sag(a + bend))
        hipc, legs = standing()
        primary, peak_used = a / D2R, peak

    elif ex == "shoulder_abduction":
        peak = lerp(70, 35, sev) if err == "limited_range" else rng.uniform(90, 150)
        a = p * peak * D2R
        bend = lerp(35, 85, sev) * D2R if err == "elbow_bent" else 0.0
        phi = lerp(25, 55, sev) * D2R if err == "arm_forward_drift" else 0.0
        if err == "trunk_lean_side":
            lean_s = -sgn * lerp(8, 20, sev) * D2R * p
        if err == "shoulder_shrug":
            shrug[side] = lerp(0.025, 0.06, sev)
        d_up = v3(sgn * math.sin(a) * math.cos(phi), -math.cos(a), math.sin(a) * math.sin(phi))
        a2 = a + bend
        d_fo = v3(sgn * math.sin(a2) * math.cos(phi), -math.cos(a2), math.sin(a2) * math.sin(phi))
        arms[side] = (d_up, d_fo)
        hipc, legs = standing()
        primary, peak_used = a / D2R, peak

    elif ex == "elbow_flexion":
        peak = lerp(75, 40, sev) if err == "limited_range" else rng.uniform(125, 150)
        b = p * peak * D2R
        drift = lerp(18, 40, sev) * D2R * p if err == "upper_arm_drift" else 0.0
        if err == "trunk_swing":
            lean_f = -lerp(8, 20, sev) * D2R * p
        arms[side] = (sag(drift), sag(drift + b))
        hipc, legs = standing()
        primary, peak_used = b / D2R, peak

    elif ex == "hip_abduction":
        peak = lerp(18, 8, sev) if err == "limited_range" else rng.uniform(28, 42)
        a = p * peak * D2R
        phi = lerp(25, 55, sev) * D2R if err == "leg_forward_drift" else 0.0
        if err == "trunk_lean_side":
            lean_s = -sgn * lerp(8, 18, sev) * D2R * p
        hipc, legs = standing()
        d = v3(sgn * math.sin(a) * math.cos(phi), -math.cos(a), math.sin(a) * math.sin(phi))
        hip = legs[side][0]
        knee = hip + B.thigh * d
        ankle = knee + B.shank * d
        legs[side] = (hip, knee, ankle)
        primary, peak_used = a / D2R, peak

    elif ex == "squat":
        peak = lerp(50, 28, sev) if err == "shallow_depth" else rng.uniform(80, 110)
        k = p * peak * D2R
        s_lean = 0.35 * k
        phi = s_lean - k
        k_deg = k / D2R
        lean_f = (5 + 0.25 * k_deg) * D2R
        if err == "excessive_forward_lean":
            lean_f += lerp(18, 35, sev) * D2R
        shift = 0.0
        if err == "lateral_weight_shift":
            shift = float(rng.choice([-1.0, 1.0])) * lerp(0.05, 0.11, sev) * p
        stance = B.hip_w * rng.uniform(1.4, 1.8)
        ax = stance / 2
        kz, ky = B.shank * math.sin(s_lean), B.shank * math.cos(s_lean)
        hz, hyy = kz + B.thigh * math.sin(phi), ky + B.thigh * math.cos(phi)
        legs = {}
        for sd, s in (("left", 1.0), ("right", -1.0)):
            ankle = v3(s * ax, 0, 0)
            hip = v3(s * lx + shift, hyy, hz)
            kx = s * ax + 0.4 * (hip[0] - s * ax)
            if err == "knee_valgus":
                kx -= s * lerp(0.04, 0.10, sev) * p
            legs[sd] = (hip, v3(kx, ky, kz), ankle)
        hipc = v3(shift, hyy, hz)
        fwd = sag(80 * D2R)
        arms = {"left": (fwd, fwd), "right": (fwd, fwd)}
        primary, peak_used = k_deg, peak

    elif ex == "marching":
        peak = lerp(42, 20, sev) if err == "low_knee_lift" else rng.uniform(70, 100)
        a = p * peak * D2R
        if err == "trunk_lean_back":
            lean_f = -lerp(10, 24, sev) * D2R * p
        if err == "torso_sway":
            lean_s = float(rng.choice([-1.0, 1.0])) * lerp(8, 18, sev) * D2R * p
        hipc, legs = standing()
        hip = legs[side][0]
        knee = hip + B.thigh * sag(a)
        ankle = knee + B.shank * sag(-rng.uniform(0, 8) * D2R)
        legs[side] = (hip, knee, ankle)
        sw = 22 * D2R * p
        arms[other] = (sag(sw), sag(sw + 10 * D2R))
        arms[side] = (sag(-12 * D2R * p), sag(-12 * D2R * p))
        primary, peak_used = a / D2R, peak

    elif ex == "seated_knee_extension":
        peak = lerp(45, 25, sev) if err == "incomplete_extension" else rng.uniform(70, 88)
        e = p * peak * D2R
        t_lift = lerp(10, 25, sev) * D2R * p if err == "thigh_lift" else 0.0
        if err == "trunk_lean_back":
            lean_f = -lerp(10, 25, sev) * D2R * p
        sy = B.shank  # hip height above floor (knee height)
        hipc = v3(0, sy, 0)
        legs = {}
        for sd, s in (("left", 1.0), ("right", -1.0)):
            hip = v3(s * lx, sy, 0)
            if sd == side:
                td = v3(0, math.sin(t_lift), math.cos(t_lift))
                knee = hip + B.thigh * td
                ankle = knee + B.shank * sag(e)
            else:
                knee = hip + v3(0, 0, B.thigh)
                ankle = knee + v3(0, -B.shank, 0)
            legs[sd] = (hip, knee, ankle)
        arms = {"left": (sag(10 * D2R), sag(80 * D2R)), "right": (sag(10 * D2R), sag(80 * D2R))}
        chair = {"seat_y": sy - 0.06}
        primary, peak_used = e / D2R, peak
    else:
        raise ValueError(ex)

    # ----- upper body assembly -----
    R = rz(-lean_s) @ rx(lean_f)
    up = R @ v3(0, 1, 0)
    xax = R @ v3(1, 0, 0)
    sc = hipc + up * B.torso
    pts: dict[str, np.ndarray] = {}
    for sd, s in (("left", 1.0), ("right", -1.0)):
        sh = sc + xax * (s * B.shoulder_w / 2) + v3(0, shrug[sd], 0)
        u_dir, f_dir = arms[sd]
        el = sh + B.upper_arm * (R @ u_dir)
        wr = el + B.forearm * (R @ f_dir)
        hip, knee, ankle = legs[sd]
        pts[f"{sd}_shoulder"], pts[f"{sd}_elbow"], pts[f"{sd}_wrist"] = sh, el, wr
        pts[f"{sd}_hip"], pts[f"{sd}_knee"], pts[f"{sd}_ankle"] = hip, knee, ankle
        pts[f"{sd}_toe"] = ankle + v3(0, -0.02, 0.15 * B.shank / 0.41)
    head_c = sc + R @ v3(0, 0.07 + B.head_r, 0.01)
    pts["head_center"] = head_c
    pts["head_top"] = head_c + up * B.head_r
    pts["nose"] = head_c + R @ v3(0, -0.01, B.head_r * 0.95)
    pts["neck"] = sc
    pts["hip_center"] = hipc
    meta = {"primary_angle_deg": float(primary), "peak_angle_deg": float(peak_used),
            "chair": chair, "lean_forward_deg": float(lean_f / D2R), "lean_side_deg": float(lean_s / D2R)}
    return pts, meta


# --------------------------------------------------------------------------
# camera + rendering
# --------------------------------------------------------------------------
def make_background(rng: np.random.Generator, W: int, H: int):
    kind = str(rng.choice(["solid", "gradient", "room", "clutter", "texture"],
                          p=[0.15, 0.2, 0.3, 0.2, 0.15]))

    def rc(lo=60, hi=230):
        return rng.integers(lo, hi, 3).astype(float)

    if kind == "solid":
        arr = np.ones((H, W, 3)) * rc()
    elif kind in ("gradient", "clutter"):
        c1, c2 = rc(), rc()
        t = np.linspace(0, 1, H)[:, None, None]
        arr = np.broadcast_to(c1 * (1 - t) + c2 * t, (H, W, 3)).copy()
    elif kind == "room":
        wall, floor = rc(120, 235), rc(60, 170)
        arr = np.ones((H, W, 3)) * wall
        hz = int(H * rng.uniform(0.55, 0.82))
        arr[hz:] = floor
        arr *= (np.linspace(rng.uniform(0.9, 1.0), rng.uniform(0.85, 1.05), W))[None, :, None]
    else:  # texture
        small = rng.random((max(2, H // 24), max(2, W // 24), 3)) * 255
        tex = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((W, H), Image.BICUBIC), dtype=float)
        arr = 0.25 * tex + 0.75 * rc()
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if kind == "room":
        for _ in range(int(rng.integers(0, 4))):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            d.rectangle([x0, y0, x0 + int(rng.integers(20, W // 2)), y0 + int(rng.integers(20, H // 2))],
                        fill=tuple(int(v) for v in rc(40, 200)))
    if kind == "clutter":
        for _ in range(int(rng.integers(4, 12))):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            x1, y1 = x0 + int(rng.integers(10, W // 2)), y0 + int(rng.integers(10, H // 2))
            col = tuple(int(v) for v in rc(40, 220))
            if rng.random() < 0.5:
                d.rectangle([x0, y0, x1, y1], fill=col)
            else:
                d.ellipse([x0, y0, x1, y1], fill=col)
    if kind in ("clutter", "room") and rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(1, 3))))
    return img, kind


def draw_cap(d, p1, p2, r1, r2, color):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L > 1e-6:
        nx, ny = -dy / L, dx / L
        d.polygon([(x1 + nx * r1, y1 + ny * r1), (x2 + nx * r2, y2 + ny * r2),
                   (x2 - nx * r2, y2 - ny * r2), (x1 - nx * r1, y1 - ny * r1)], fill=color)
    d.ellipse([x1 - r1, y1 - r1, x1 + r1, y1 + r1], fill=color)
    d.ellipse([x2 - r2, y2 - r2, x2 + r2, y2 + r2], fill=color)


def render_sample(pts: dict, meta: dict, B: Body, app: dict, cam: dict,
                  W: int, H: int, rng: np.random.Generator, ss: int = 2):
    """Project the 3D pose, draw the figure, return (PIL image, joint records, info)."""
    names = list(pts.keys())
    P = np.array([pts[n] for n in names])
    cy = float(P[:, 1].mean())
    M = rz(cam["roll"] * D2R) @ rx(cam["pitch"] * D2R) @ ry(cam["yaw"] * D2R)

    def cam_coords(X: np.ndarray):
        Q = (X - v3(0, cy, 0)) @ M.T
        depth = np.maximum(cam["dist"] - Q[:, 2], 0.3)
        return Q[:, 0] / depth, -Q[:, 1] / depth, depth

    u0, v0, d0 = cam_coords(P)
    rad0 = 0.07 / d0
    umin, umax = float((u0 - rad0).min()), float((u0 + rad0).max())
    vmin, vmax = float((v0 - rad0).min()), float((v0 + rad0).max())
    bw0, bh0 = umax - umin, vmax - vmin
    f = min(cam["frac"] * H / bh0, 0.92 * W / bw0)
    margin = 0.03 * min(W, H)
    ox = W / 2 - f * (umin + umax) / 2
    oy = H / 2 - f * (vmin + vmax) / 2
    sx = max(0.0, (W - f * bw0) / 2 - margin)
    sy = max(0.0, (H - f * bh0) / 2 - margin)
    ox += rng.uniform(-sx, sx)
    oy += rng.uniform(-sy, sy)
    if cam["crop"]:
        ox += rng.uniform(-0.25, 0.25) * W
        oy += rng.uniform(-0.12, 0.12) * H

    def project(X: np.ndarray):
        u, v, dep = cam_coords(X)
        return f * u + ox, f * v + oy, dep

    U, V, DP = project(P)
    idx = {n: i for i, n in enumerate(names)}

    def px(n):  # supersampled pixel position
        return (U[idx[n]] * ss, V[idx[n]] * ss)

    def rp(n, r_m):  # radius in supersampled pixels for a sphere of r_m metres at point n
        return r_m * f / DP[idx[n]] * ss

    items = []  # (depth, draw function)
    bld = B.build
    skin, hair, shirt, pants, shoes = (app[k] for k in ("skin", "hair", "shirt", "pants", "shoes"))
    long_sleeve = app["sleeve"] == "long"
    long_legs = app["legwear"] == "long"

    # ground shadow
    gx = (pts["left_ankle"][0] + pts["right_ankle"][0]) / 2
    gz = (pts["left_ankle"][2] + pts["right_ankle"][2]) / 2
    gu, gv, gd = project(np.array([[gx, 0.0, gz]]))
    sr = 0.35 * f / gd[0] * ss
    sq = max(0.12, abs(math.sin(cam["pitch"] * D2R)) + 0.15)
    shadow_box = [gu[0] * ss - sr, gv[0] * ss - sr * sq, gu[0] * ss + sr, gv[0] * ss + sr * sq]
    items.append((1e3, lambda d, b=shadow_box: d.ellipse(b, fill=(0, 0, 0, 55))))

    # chair
    if meta["chair"] is not None:
        sy_ = meta["chair"]["seat_y"]
        c_col = (140, 100, 60) if rng.random() < 0.5 else (90, 90, 100)
        c_dark = tuple(int(v * 0.7) for v in c_col)
        seat3 = np.array([[-0.25, sy_, -0.22], [0.25, sy_, -0.22], [0.25, sy_, 0.27], [-0.25, sy_, 0.27]])
        back3 = np.array([[-0.23, sy_, -0.24], [0.23, sy_, -0.24], [0.23, sy_ + 0.5, -0.24], [-0.23, sy_ + 0.5, -0.24]])
        for pts3, col, bias in ((seat3, c_col, 0.25), (back3, c_dark, 0.0)):
            cu, cv, cd = project(pts3)
            poly = [(float(a * ss), float(b * ss)) for a, b in zip(cu, cv)]
            items.append((float(cd.mean()) + bias, lambda d, pl=poly, c=col: d.polygon(pl, fill=c)))
        for lx_, lz_ in ((-0.23, -0.2), (0.23, -0.2), (-0.23, 0.25), (0.23, 0.25)):
            lu, lv, ld = project(np.array([[lx_, sy_, lz_], [lx_, 0.0, lz_]]))
            p1, p2 = (lu[0] * ss, lv[0] * ss), (lu[1] * ss, lv[1] * ss)
            rr = 0.02 * f / ld.mean() * ss
            items.append((float(ld.mean()) + 0.3, lambda d, a=p1, b=p2, r=rr, c=c_dark: draw_cap(d, a, b, r, r, c)))

    def add_cap(a, b, ra, rb, color, bias=0.0):
        dep = float((DP[idx[a]] + DP[idx[b]]) / 2 + bias)
        pa, pb = px(a), px(b)
        r1, r2 = rp(a, ra), rp(b, rb)
        items.append((dep, lambda d, pa=pa, pb=pb, r1=r1, r2=r2, c=color: draw_cap(d, pa, pb, r1, r2, c)))

    # torso (tapered capsule from neck to hip centre; width follows projected shoulder/hip width)
    sh_half = math.hypot(U[idx["left_shoulder"]] - U[idx["right_shoulder"]],
                         V[idx["left_shoulder"]] - V[idx["right_shoulder"]]) / 2
    hp_half = math.hypot(U[idx["left_hip"]] - U[idx["right_hip"]],
                         V[idx["left_hip"]] - V[idx["right_hip"]]) / 2
    d_sc, d_hc = DP[idx["neck"]], DP[idx["hip_center"]]
    r_top = max(sh_half, 0.10 * bld * f / d_sc) + 0.03 * f / d_sc
    r_bot = max(hp_half * 1.15, 0.095 * bld * f / d_hc) + 0.02 * f / d_hc
    items.append(((d_sc + d_hc) / 2 + 0.05,
                  lambda d, a=px("neck"), b=px("hip_center"), r1=r_top * ss, r2=r_bot * ss, c=shirt:
                  draw_cap(d, a, b, r1, r2, c)))
    # pelvis block in pants colour (lower third of torso)
    mid = ((U[idx["neck"]] * 1 + 2 * U[idx["hip_center"]]) / 3 * ss, (V[idx["neck"]] + 2 * V[idx["hip_center"]]) / 3 * ss)
    items.append(((d_sc + d_hc) / 2 + 0.049,
                  lambda d, a=mid, b=px("hip_center"), r1=r_bot * ss * 1.0, r2=r_bot * ss, c=pants:
                  draw_cap(d, a, b, r1, r2, c)))
    # neck + head
    add_cap("neck", "head_center", 0.045 * bld, 0.045 * bld, skin, bias=0.0)
    hp, hr = px("head_center"), rp("head_center", B.head_r)
    nose_p = px("nose")
    facing = DP[idx["nose"]] <= DP[idx["head_center"]] + 0.005

    def draw_head(d, hp=hp, hr=hr, nose_p=nose_p, facing=facing, hair=hair, skin=skin):
        d.ellipse([hp[0] - hr * 1.04, hp[1] - hr * 1.04, hp[0] + hr * 1.04, hp[1] + hr * 1.04], fill=hair)
        if facing:
            fc = (hp[0] + (nose_p[0] - hp[0]) * 0.35, hp[1] + (nose_p[1] - hp[1]) * 0.35)
            fr = hr * 0.9
            d.ellipse([fc[0] - fr, fc[1] - fr, fc[0] + fr, fc[1] + fr], fill=skin)
            dark = tuple(int(v * 0.7) for v in skin)
            nr = hr * 0.14
            d.ellipse([nose_p[0] - nr, nose_p[1] - nr, nose_p[0] + nr, nose_p[1] + nr], fill=dark)
    items.append((float(DP[idx["head_center"]]) - 0.1, draw_head))

    for sd in ("left", "right"):
        sh, el, wr = f"{sd}_shoulder", f"{sd}_elbow", f"{sd}_wrist"
        hi, kn, an, to = f"{sd}_hip", f"{sd}_knee", f"{sd}_ankle", f"{sd}_toe"
        add_cap(sh, el, 0.044 * bld, 0.038 * bld, shirt, bias=-0.02)
        add_cap(el, wr, 0.036 * bld, 0.030 * bld, shirt if long_sleeve else skin, bias=-0.02)
        add_cap(wr, wr, 0.04, 0.04, skin, bias=-0.02)
        add_cap(sh, sh, 0.055 * bld, 0.055 * bld, shirt, bias=-0.02)
        add_cap(hi, kn, 0.075 * bld, 0.060 * bld, pants)
        add_cap(kn, an, 0.056 * bld, 0.044 * bld, pants if long_legs else skin)
        add_cap(an, to, 0.038, 0.034, shoes, bias=-0.01)

    items.sort(key=lambda t: -t[0])
    fg = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    dr = ImageDraw.Draw(fg)
    for _, fn in items:
        fn(dr)
    fg = fg.resize((W, H), Image.LANCZOS)

    bg, bg_kind = make_background(rng, W, H)
    img = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")

    # post effects (recorded in annotations)
    fx = {"brightness": float(rng.uniform(0.75, 1.25)), "contrast": float(rng.uniform(0.8, 1.2)),
          "blur_radius": float(rng.uniform(0.3, 1.2)) if rng.random() < 0.4 else 0.0,
          "noise_sigma": float(rng.uniform(0, 6)),
          "jpeg_quality": int(rng.integers(55, 96)) if rng.random() < 0.3 else None}
    img = ImageEnhance.Brightness(img).enhance(fx["brightness"])
    img = ImageEnhance.Contrast(img).enhance(fx["contrast"])
    if fx["blur_radius"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(fx["blur_radius"]))
    arr = np.asarray(img, dtype=float)
    arr = np.clip(arr + rng.normal(0, fx["noise_sigma"], arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if fx["jpeg_quality"] is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=fx["jpeg_quality"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # ----- joint annotations -----
    A = np.array([U[idx["neck"]], V[idx["neck"]]])
    Bp = np.array([U[idx["hip_center"]], V[idx["hip_center"]]])
    ab = Bp - A
    ab2 = float(ab @ ab)
    joints = []
    for jn in JOINT_NAMES:
        i = idx[jn]
        u, v, dep = float(U[i]), float(V[i]), float(DP[i])
        inside = 0 <= u < W and 0 <= v < H
        vis = 1.0
        occluded = False
        if jn == "nose":
            occluded = DP[i] > DP[idx["head_center"]] + 0.02
        else:
            t = float(np.clip(((np.array([u, v]) - A) @ ab) / ab2, 0, 1)) if ab2 > 1e-9 else 0.0
            closest = A + t * ab
            dist = float(np.linalg.norm(np.array([u, v]) - closest))
            r = lerp(r_top, r_bot, t)
            d_ax = lerp(float(d_sc), float(d_hc), t)
            occluded = dist < 0.85 * r and dep > d_ax + 0.08
        if not inside:
            vis = 0.0
        elif occluded:
            vis = 0.3
        else:
            vis = float(rng.uniform(0.9, 1.0))
        w3 = pts[jn]
        joints.append({"name": jn, "mediapipe_index": MEDIAPIPE_INDEX[jn],
                       "x_px": round(u, 2), "y_px": round(v, 2),
                       "x_norm": round(u / W, 5), "y_norm": round(v / H, 5),
                       "visibility": round(vis, 3),
                       "world_xyz_m": [round(float(w3[0]), 4), round(float(w3[1]), 4), round(float(w3[2]), 4)]})
    return img, joints, {"background": bg_kind, "effects": fx}


# --------------------------------------------------------------------------
# dataset driver
# --------------------------------------------------------------------------
def parse_args():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Generate the POSTURE synthetic exercise dataset.")
    ap.add_argument("--out", type=Path, default=root / "dataset" / "synthetic", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed (same seed => identical dataset)")
    ap.add_argument("--num-subjects", type=int, default=60, help="number of synthetic people")
    ap.add_argument("--clips-per-subject", type=int, default=35, help="clips (repetition settings) per subject")
    ap.add_argument("--frames-per-clip", type=int, default=1, help="images rendered per clip (same camera/person)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--max-yaw", type=float, default=90.0, help="camera yaw sampled in [-max, +max] degrees")
    ap.add_argument("--phase-min", type=float, default=0.75,
                    help="images are sampled at movement progress in [phase-min, 1]; 1 = peak of the rep")
    ap.add_argument("--incorrect-fraction", type=float, default=0.5)
    ap.add_argument("--crop-fraction", type=float, default=0.03, help="share of clips partly cropped by the frame")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true", help="replace previously generated images/annotations in --out")
    return ap.parse_args()


def assign_splits(n: int, val_frac: float, test_frac: float, rng: np.random.Generator) -> dict[int, str]:
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise SystemExit("Not enough subjects for a train/val/test split; increase --num-subjects.")
    order = rng.permutation(n)
    out = {}
    for rank, s in enumerate(order):
        out[int(s)] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return out


def main():
    a = parse_args()
    out: Path = a.out
    managed = ["images", "annotations", "preview"]
    if (out / "images").exists() and any((out / "images").iterdir()):
        if not a.overwrite:
            sys.exit(f"{out}/images already contains data. Re-run with --overwrite to replace it.")
    for sub in managed:
        shutil.rmtree(out / sub, ignore_errors=True)
    for f in ("dataset_summary.json", "dataset_summary.txt"):
        (out / f).unlink(missing_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    master = np.random.default_rng(a.seed)
    subjects = [make_subject(i, np.random.default_rng([a.seed, 1, i])) for i in range(a.num_subjects)]
    split_of = assign_splits(a.num_subjects, a.val_fraction, a.test_fraction, master)
    ex_list = list(EXERCISES.keys())

    records = []
    counter = 0
    total = a.num_subjects * a.clips_per_subject * a.frames_per_clip
    for subj in subjects:
        split = split_of[subj["index"]]
        for k in range(a.clips_per_subject):
            crng = np.random.default_rng([a.seed, 2, subj["index"], k])
            ex = ex_list[(k + subj["index"]) % len(ex_list)]
            incorrect = crng.random() < a.incorrect_fraction
            err = str(crng.choice(EXERCISES[ex])) if incorrect else "none"
            sev = float(crng.uniform(0.0, 1.0)) if incorrect else 0.0
            side = str(crng.choice(["left", "right"]))
            cam = {"yaw": float(crng.uniform(-a.max_yaw, a.max_yaw)),
                   "pitch": float(np.clip(crng.normal(0, 4), -10, 10)),
                   "roll": float(np.clip(crng.normal(0, 1.5), -5, 5)),
                   "dist": float(crng.uniform(2.2, 5.0)),
                   "frac": float(crng.uniform(0.55, 0.88)),
                   "crop": bool(crng.random() < a.crop_fraction)}
            phases = sorted(float(x) for x in crng.uniform(a.phase_min, 1.0, a.frames_per_clip))
            clip_id = f"c{subj['index']:03d}_{k:03d}"
            for j, ph in enumerate(phases):
                pts, meta = build_pose(ex, subj["body"], ph, err, sev, side, crng)
                img, joints, info = render_sample(pts, meta, subj["body"], subj["appearance"], cam,
                                                  a.width, a.height, crng)
                sample_id = f"s{counter:06d}"
                rel = Path("images") / split / ex / f"{sample_id}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save(out / rel, compress_level=6)
                records.append({
                    "sample_id": sample_id, "clip_id": clip_id, "frame_in_clip": j,
                    "image_path": rel.as_posix(), "split": split, "subject_id": subj["subject_id"],
                    "exercise": ex, "active_side": side if ex != "squat" else "both",
                    "form_label": "incorrect" if incorrect else "correct",
                    "is_correct": 0 if incorrect else 1,
                    "error_type": err, "error_severity": round(sev, 3),
                    "movement_progress": round(ph, 3),
                    "primary_angle_deg": round(meta["primary_angle_deg"], 2),
                    "target_peak_angle_deg": round(meta["peak_angle_deg"], 2),
                    "camera_yaw_deg": round(cam["yaw"], 2), "camera_pitch_deg": round(cam["pitch"], 2),
                    "camera_roll_deg": round(cam["roll"], 2), "camera_distance_m": round(cam["dist"], 2),
                    "cropped_by_frame": cam["crop"],
                    "image_width": a.width, "image_height": a.height,
                    "background": info["background"], "effects": info["effects"],
                    "appearance": {k2: (list(v) if isinstance(v, tuple) else v)
                                   for k2, v in subj["appearance"].items()},
                    "body": {k2: round(float(v), 4) for k2, v in subj["body"].__dict__.items()},
                    "joints": joints,
                })
                counter += 1
        print(f"  {subj['subject_id']} ({split}) done  [{counter}/{total} images]", flush=True)

    # ---- write annotations ----
    ann_dir = out / "annotations"
    meta_block = {"schema_version": SCHEMA_VERSION, "seed": a.seed,
                  "joint_names": JOINT_NAMES, "mediapipe_index": MEDIAPIPE_INDEX,
                  "coordinate_notes": {
                      "pixels": "origin top-left, x right, y down",
                      "world_xyz_m": "+x subject's left, +y up, +z forward (direction subject faces), metres",
                      "camera_yaw_deg": "positive yaw: subject's front turns toward the RIGHT of the image; 0 = facing camera",
                      "camera_pitch_deg": "positive = camera above subject looking down",
                      "visibility": "heuristic: 0 = outside frame, 0.3 = hidden behind torso/head, ~0.9-1 = visible",
                  },
                  "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    with open(ann_dir / "annotations.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta_block, "samples": records}, fh, indent=1)

    base_cols = ["sample_id", "clip_id", "frame_in_clip", "image_path", "split", "subject_id", "exercise",
                 "active_side", "form_label", "is_correct", "error_type", "error_severity", "movement_progress",
                 "primary_angle_deg", "target_peak_angle_deg", "camera_yaw_deg", "camera_pitch_deg",
                 "camera_roll_deg", "camera_distance_m", "cropped_by_frame", "image_width", "image_height"]
    joint_cols = [f"{j}_{s}" for j in JOINT_NAMES for s in ("x", "y", "v")]
    with open(ann_dir / "annotations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(base_cols + joint_cols)
        for r in records:
            row = [r[c] for c in base_cols]
            for jt in r["joints"]:
                row += [jt["x_norm"], jt["y_norm"], jt["visibility"]]
            w.writerow(row)
    for sp in ("train", "val", "test"):
        with open(ann_dir / f"{sp}.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(r["sample_id"] for r in records if r["split"] == sp) + "\n")

    write_summary(out, records, subjects, split_of, a, time.time() - t0)
    print(f"\nDone: {len(records)} images in {time.time() - t0:.1f} s -> {out}")


def yaw_bin(y: float) -> str:
    lo = int(math.floor(abs(y) / 30) * 30)
    lo = min(lo, 60)
    return f"{lo}-{lo + 30} deg" if lo < 60 else "60-90+ deg"


def write_summary(out: Path, recs: list, subjects: list, split_of: dict, a, seconds: float):
    def count(key, flt=None):
        c = Counter()
        for r in recs:
            if flt is None or flt(r):
                c[r[key]] += 1
        return dict(sorted(c.items()))

    splits = ("train", "val", "test")
    summary = {
        "total_images": len(recs), "seed": a.seed, "seconds": round(seconds, 1),
        "subjects": {sp: sum(1 for s in split_of.values() if s == sp) for sp in splits},
        "images_per_split": count("split"),
        "images_per_exercise": count("exercise"),
        "images_per_form_label": count("form_label"),
        "error_type_counts": count("error_type", lambda r: r["error_type"] != "none"),
        "per_split_form_label": {sp: count("form_label", lambda r, sp=sp: r["split"] == sp) for sp in splits},
        "per_split_exercise": {sp: count("exercise", lambda r, sp=sp: r["split"] == sp) for sp in splits},
        "yaw_bins": dict(sorted(Counter(yaw_bin(r["camera_yaw_deg"]) for r in recs).items())),
        "cropped_by_frame": sum(1 for r in recs if r["cropped_by_frame"]),
        "background_types": count("background"),
        "subject_overlap_between_splits": 0,
    }
    seen = defaultdict(set)
    for r in recs:
        seen[r["subject_id"]].add(r["split"])
    summary["subject_overlap_between_splits"] = sum(1 for v in seen.values() if len(v) > 1)
    vis = [j["visibility"] for r in recs for j in r["joints"]]
    summary["joint_visibility"] = {"visible(>0.5)": sum(v > 0.5 for v in vis), "hidden_by_body(0.3)": sum(v == 0.3 for v in vis),
                                   "outside_frame(0)": sum(v == 0.0 for v in vis)}
    with open(out / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    L = []
    L.append("POSTURE synthetic dataset summary")
    L.append("=" * 40)
    L.append(f"Total images: {summary['total_images']}   seed: {a.seed}   image size: {a.width}x{a.height}")
    L.append(f"Subjects per split: {summary['subjects']}   (overlap between splits: {summary['subject_overlap_between_splits']})")
    for title, key in (("Images per split", "images_per_split"), ("Images per exercise", "images_per_exercise"),
                       ("Form label", "images_per_form_label"), ("Error types (incorrect samples)", "error_type_counts"),
                       ("Camera yaw |deg|", "yaw_bins"), ("Backgrounds", "background_types"),
                       ("Joint visibility (all joints)", "joint_visibility")):
        L.append(f"\n{title}:")
        for k, v in summary[key].items():
            L.append(f"  {k:<28}{v}")
    L.append("\nPer-split form label:")
    for sp in splits:
        L.append(f"  {sp:<8}{summary['per_split_form_label'][sp]}")
    L.append(f"\nImages partly cropped by frame: {summary['cropped_by_frame']}")
    L.append("\nWHAT THIS DATA IS / IS NOT")
    L.append("- Stylised rendered figures with exact joint annotations; useful for building and testing the")
    L.append("  pipeline (data loading, normalisation, angle maths, yaw handling, model plumbing).")
    L.append("- NOT real people, NOT real webcam footage, NOT MediaPipe output. Real-world accuracy must be")
    L.append("  measured on real recordings (your volunteer data) and is not established by this dataset.")
    L.append("- Form labels come from generator parameters, not clinical judgement. Some incorrect samples are")
    L.append("  visually indistinguishable from correct ones at some camera angles.")
    (out / "dataset_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()