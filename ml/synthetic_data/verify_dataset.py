#!/usr/bin/env python3
"""
POSTURE - Phase 2: verify the generated synthetic dataset and draw a preview grid.

Checks (each prints PASS or FAIL):
  1. annotations.json and annotations.csv exist and agree on the number of samples
  2. every image file exists and has the size recorded in the annotations
  3. every sample has exactly 13 joints with finite coordinates
  4. visibility is consistent with the frame (outside frame => 0, inside => > 0)
  5. labels are consistent (correct <=> error_type 'none'; errors belong to their exercise)
  6. no subject appears in more than one split (no person/appearance leakage)
  7. sample ids are unique and split lists match the annotations
  8. joints annotated inside a plausible bounding box (sanity check on projection)

It also writes <dataset>/preview/preview_grid.png with the skeleton drawn over
24 random images so you can LOOK at whether annotations sit on the body.

Usage (from the repository root):
    python ml/synthetic_data/verify_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=root / "dataset" / "synthetic")
    ap.add_argument("--preview-count", type=int, default=24)
    a = ap.parse_args()
    ds: Path = a.dataset

    results = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    ann_json = ds / "annotations" / "annotations.json"
    ann_csv = ds / "annotations" / "annotations.csv"
    if not ann_json.exists():
        sys.exit(f"Missing {ann_json}. Run generate_dataset.py first.")
    data = json.loads(ann_json.read_text(encoding="utf-8"))
    recs = data["samples"]
    valid_errors = None
    # error vocabulary is imported from the generator so both files stay in sync
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_dataset import EXERCISES, JOINT_NAMES  # noqa: E402
    valid_errors = EXERCISES

    with open(ann_csv, newline="", encoding="utf-8") as fh:
        n_csv = sum(1 for _ in csv.reader(fh)) - 1
    check("JSON and CSV have the same number of samples", n_csv == len(recs), f"json={len(recs)} csv={n_csv}")

    missing, bad_size = [], []
    for r in recs:
        p = ds / r["image_path"]
        if not p.exists():
            missing.append(r["image_path"])
            continue
        with Image.open(p) as im:
            if im.size != (r["image_width"], r["image_height"]):
                bad_size.append(r["image_path"])
    check("all image files exist", not missing, f"{len(missing)} missing")
    check("image sizes match annotations", not bad_size, f"{len(bad_size)} mismatched")

    bad_joints = 0
    bad_vis = 0
    for r in recs:
        js = r["joints"]
        if [j["name"] for j in js] != JOINT_NAMES:
            bad_joints += 1
            continue
        for j in js:
            vals = [j["x_px"], j["y_px"], j["visibility"]]
            if not all(math.isfinite(v) for v in vals):
                bad_joints += 1
                break
            inside = 0 <= j["x_px"] < r["image_width"] and 0 <= j["y_px"] < r["image_height"]
            if (not inside and j["visibility"] != 0) or (inside and j["visibility"] <= 0):
                bad_vis += 1
    check("every sample has the 13 joints with finite values", bad_joints == 0, f"{bad_joints} bad")
    check("visibility consistent with frame", bad_vis == 0, f"{bad_vis} inconsistent joints")

    bad_lab = 0
    for r in recs:
        ok = (r["form_label"] == "correct") == (r["error_type"] == "none") == (r["is_correct"] == 1)
        if r["error_type"] != "none" and r["error_type"] not in valid_errors[r["exercise"]]:
            ok = False
        bad_lab += (not ok)
    check("labels consistent", bad_lab == 0, f"{bad_lab} inconsistent")

    seen = defaultdict(set)
    for r in recs:
        seen[r["subject_id"]].add(r["split"])
    leaked = [s for s, v in seen.items() if len(v) > 1]
    check("no subject appears in more than one split", not leaked, f"{len(leaked)} leaked")

    ids = [r["sample_id"] for r in recs]
    split_files_ok = True
    for sp in ("train", "val", "test"):
        f = ds / "annotations" / f"{sp}.txt"
        listed = set(f.read_text().split()) if f.exists() else None
        expected = {r["sample_id"] for r in recs if r["split"] == sp}
        if listed != expected:
            split_files_ok = False
    check("sample ids unique", len(ids) == len(set(ids)))
    check("train/val/test list files match annotations", split_files_ok)

    odd = 0
    for r in recs:
        if r["cropped_by_frame"]:
            continue
        xs = [j["x_px"] for j in r["joints"]]
        ys = [j["y_px"] for j in r["joints"]]
        if min(xs) < -1 or max(xs) > r["image_width"] + 1 or min(ys) < -1 or max(ys) > r["image_height"] + 1:
            odd += 1
    check("non-cropped samples have all joints inside the frame", odd == 0, f"{odd} outside")

    # ---- preview grid ----
    rng = np.random.default_rng(0)
    pick = rng.choice(len(recs), size=min(a.preview_count, len(recs)), replace=False)
    cols = 6
    rows = math.ceil(len(pick) / cols)
    w, h = recs[0]["image_width"], recs[0]["image_height"]
    grid = Image.new("RGB", (cols * w, rows * (h + 14)), (255, 255, 255))
    for n, i in enumerate(pick):
        r = recs[int(i)]
        im = Image.open(ds / r["image_path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        pos = {j["name"]: (j["x_px"], j["y_px"], j["visibility"]) for j in r["joints"]}
        for s, e in BONES:
            d.line([pos[s][:2], pos[e][:2]], fill=(0, 255, 0), width=1)
        for name, (x, y, v) in pos.items():
            col = (255, 0, 0) if v > 0.5 else (255, 200, 0)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col, width=2)
        cx, cy = (n % cols) * w, (n // cols) * (h + 14)
        grid.paste(im, (cx, cy))
        label = f"{r['exercise'][:14]} {'OK' if r['is_correct'] else r['error_type'][:10]} y{r['camera_yaw_deg']:.0f}"
        ImageDraw.Draw(grid).text((cx + 2, cy + h + 1), label, fill=(0, 0, 0))
    (ds / "preview").mkdir(exist_ok=True)
    grid.save(ds / "preview" / "preview_grid.png")
    print(f"\nPreview written: {ds / 'preview' / 'preview_grid.png'}")

    if all(results):
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()