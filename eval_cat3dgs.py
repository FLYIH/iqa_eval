#!/usr/bin/env python
"""
Evaluate CAT3DGSPro_organized (methods: 3DCR, Orig) and HACplus+CR
against the shared ground-truth reference views, using pyiqa no-reference
metrics: MUSIQ, MANIQA, CLIP-IQA, NIQE.

NOTE: this script used to also compute a whole-image, no-preprocessing FID
(pyiqa's stock FID on full-resolution images). That code was removed --
whole-image FID at our sample sizes (16-39 images/scene) is too unstable to
trust (see README.md's FID sample-size caveat), and we now have two better
alternatives instead: `eval_fid_patch.py` (HiFiC-style patch FID) and
`eval_fid_cmmd_fivecrop.py` (five-crop FID+CMMD, matches a specific paper's
protocol). The `fid` column already present in existing per_scene.csv /
summary.csv files (from before this change) is left as historical data --
this script will not regenerate or extend it.

Data layout on disk:

  CAT3DGSPro_organized/{scene}_{idx:05d}/
      reference.png     <- ground truth for this view
      3DCR@{rate}.png    <- method "3DCR" rendered at rate in {high, mid, low}
      Orig@{rate}.png    <- method "Orig" rendered at rate in {high, mid, low}

  HACplus+CR/HACplus+CR/{RateDir}/{scene}/{idx:05d}.png
      RateDir in {"High (0.002)", "Mid (0.007)", "Low (0.05)"}
      -> method "HACplus+CR" rendered at the corresponding rate

Views are index-aligned across all three methods per scene (verified: same
counts and same 00000..N indices), so the reference for
HACplus+CR/.../{scene}/00003.png is CAT3DGSPro_organized/{scene}_00003/reference.png.

Usage:
  python eval_cat3dgs.py --cat3dgs_root /work/u4546465/evaluation/CAT3DGSPro_organized \
                          --hac_root "/work/u4546465/evaluation/HACplus+CR" \
                          --out_dir /home/u4546465/iqa_eval/results \
                          --device cuda

  # quick smoke test on a couple of scenes/images before the full run:
  python eval_cat3dgs.py ... --limit_scenes bicycle,bonsai --limit_per_scene 3

Outputs (in --out_dir):
  per_image.csv    one row per (method, rate, scene, idx): musiq/maniqa/clipiqa/niqe
  per_scene.csv    one row per (method, rate, scene): mean NR metrics (n_images, plus a
                   stale "fid" column carried over from old runs -- see NOTE above)
  summary.csv      one row per (method, rate), pooled across all scenes: mean NR metrics
                   (same stale "fid" column caveat as per_scene.csv)
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RATES = ["high", "mid", "low"]

# "External" methods: images live under <hac_root>/<subdir>/<rate_dir>/<scene>/<idx>.png
# rather than inside CAT3DGSPro_organized. Add new entries here to evaluate more methods
# without touching the rest of the script.
EXTERNAL_METHODS = {
    "HACplus+CR": {
        "subdir": "HACplus+CR",
        "rate_dirs": {"high": "High (0.002)", "mid": "Mid (0.007)", "low": "Low (0.05)"},
    },
    "HAC_ori": {
        "subdir": "HAC_ori",
        "rate_dirs": {"high": "High (0.003)", "mid": "Mid (0.01)", "low": "Low (0.06)"},
    },
}

NR_METRICS = ["musiq", "maniqa", "clipiqa", "niqe"]
SCENE_RE = re.compile(r"^(?P<scene>.+)_(?P<idx>\d+)$")


def discover_views(cat3dgs_root, limit_scenes=None, limit_per_scene=None):
    """Return dict: scene -> sorted list of idx strings (e.g. '00000'), and
    dict: (scene, idx) -> Path to reference.png"""
    cat3dgs_root = Path(cat3dgs_root)
    scenes = defaultdict(list)
    ref_paths = {}
    for d in sorted(cat3dgs_root.iterdir()):
        if not d.is_dir():
            continue
        m = SCENE_RE.match(d.name)
        if not m:
            continue
        scene, idx = m.group("scene"), m.group("idx")
        if limit_scenes and scene not in limit_scenes:
            continue
        ref = d / "reference.png"
        if not ref.exists():
            print(f"WARNING: missing reference.png in {d}", file=sys.stderr)
            continue
        scenes[scene].append(idx)
        ref_paths[(scene, idx)] = ref

    for scene in scenes:
        scenes[scene].sort()
        if limit_per_scene:
            scenes[scene] = scenes[scene][:limit_per_scene]

    valid_keys = {(scene, idx) for scene, idxs in scenes.items() for idx in idxs}
    ref_paths = {key: path for key, path in ref_paths.items() if key in valid_keys}

    return scenes, ref_paths


def cat3dgs_method_path(cat3dgs_root, scene, idx, method, rate):
    return Path(cat3dgs_root) / f"{scene}_{idx}" / f"{method}@{rate}.png"


def external_method_path(hac_root, method_cfg, scene, idx, rate):
    rate_dir = method_cfg["rate_dirs"][rate]
    return Path(hac_root) / method_cfg["subdir"] / rate_dir / scene / f"{idx}.png"


def build_records(cat3dgs_root, hac_root, scenes, methods=None):
    """Return list of dicts: {method, rate, scene, idx, path, ref_path}.
    methods: optional set of method names to restrict to (default: all)."""
    records = []
    _, ref_paths = discover_views(cat3dgs_root)
    embedded_methods = [m for m in ["3DCR", "Orig", "wdr_pw"] if methods is None or m in methods]
    external_methods = {n: c for n, c in EXTERNAL_METHODS.items() if methods is None or n in methods}

    for scene, idxs in scenes.items():
        for idx in idxs:
            ref_path = ref_paths[(scene, idx)]
            for rate in RATES:
                for method in embedded_methods:
                    p = cat3dgs_method_path(cat3dgs_root, scene, idx, method, rate)
                    if p.exists():
                        records.append(dict(method=method, rate=rate, scene=scene, idx=idx,
                                             path=p, ref_path=ref_path))
                    else:
                        print(f"WARNING: missing {p}", file=sys.stderr)
                for method_name, method_cfg in external_methods.items():
                    p = external_method_path(hac_root, method_cfg, scene, idx, rate)
                    if p.exists():
                        records.append(dict(method=method_name, rate=rate, scene=scene, idx=idx,
                                             path=p, ref_path=ref_path))
                    else:
                        print(f"WARNING: missing {p}", file=sys.stderr)
    return records


def run_nr_metrics(records, metrics, device):
    import torch
    import pyiqa
    models = {m: pyiqa.create_metric(m, device=device) for m in metrics}
    for i, r in enumerate(records):
        for m in metrics:
            with torch.no_grad():
                r[m] = models[m](str(r["path"])).item()
        if (i + 1) % 100 == 0 or i == len(records) - 1:
            print(f"  NR metrics [{i+1}/{len(records)}]")
    return records


def load_existing_rows(path, key_fields):
    """Read an existing CSV (if present) into a dict keyed by key_fields tuple -> row dict."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {tuple(row[k] for k in key_fields): row for row in reader}


def write_csvs(records, out_dir, metrics, merge=True):
    """Write per_image/per_scene/summary CSVs. If merge=True and the CSVs already
    exist, rows for methods NOT present in `records` are preserved untouched, and
    rows for methods that ARE present are replaced with the freshly computed ones
    (so re-running one method doesn't clobber results for the others).

    per_scene.csv/summary.csv no longer have FID in their schema (see module
    docstring). extrasaction="ignore" is used when writing so that preserved
    rows loaded from an older CSV (which may still carry a stale "fid" key)
    don't crash the writer -- that historical fid value is simply dropped from
    the rewritten row, the untouched original file already has it recorded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_methods = {r["method"] for r in records}

    # per_image.csv (key: method, rate, scene, idx)
    per_image_path = out_dir / "per_image.csv"
    fieldnames = ["method", "rate", "scene", "idx", "path", *metrics]
    existing = load_existing_rows(per_image_path, ["method", "rate", "scene", "idx"]) if merge else {}
    existing = {k: v for k, v in existing.items() if v["method"] not in new_methods}
    new_rows = {(r["method"], r["rate"], r["scene"], r["idx"]):
                {k: r[k] for k in fieldnames if k != "path"} | {"path": str(r["path"])}
                for r in records}
    all_rows = list(existing.values()) + list(new_rows.values())
    all_rows.sort(key=lambda r: (r["method"], r["rate"], r["scene"], r["idx"]))
    with open(per_image_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {per_image_path} ({len(all_rows)} rows)")

    # per_scene.csv  (group by method, rate, scene)
    groups = defaultdict(list)
    for r in records:
        groups[(r["method"], r["rate"], r["scene"])].append(r)

    per_scene_path = out_dir / "per_scene.csv"
    fieldnames = ["method", "rate", "scene", "n_images", *[f"{m}_mean" for m in metrics]]
    existing = load_existing_rows(per_scene_path, ["method", "rate", "scene"]) if merge else {}
    existing = {k: v for k, v in existing.items() if v["method"] not in new_methods}
    new_rows = {}
    for (method, rate, scene), rs in groups.items():
        row = {"method": method, "rate": rate, "scene": scene, "n_images": len(rs)}
        for m in metrics:
            row[f"{m}_mean"] = sum(r[m] for r in rs) / len(rs)
        new_rows[(method, rate, scene)] = row
    all_rows = list(existing.values()) + list(new_rows.values())
    all_rows.sort(key=lambda r: (r["method"], r["rate"], r["scene"]))
    with open(per_scene_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {per_scene_path} ({len(all_rows)} rows)")

    # summary.csv (group by method, rate; pooled across scenes)
    groups2 = defaultdict(list)
    for r in records:
        groups2[(r["method"], r["rate"])].append(r)

    summary_path = out_dir / "summary.csv"
    fieldnames = ["method", "rate", "n_images", *[f"{m}_mean" for m in metrics]]
    existing = load_existing_rows(summary_path, ["method", "rate"]) if merge else {}
    existing = {k: v for k, v in existing.items() if v["method"] not in new_methods}
    new_rows = {}
    for (method, rate), rs in groups2.items():
        row = {"method": method, "rate": rate, "n_images": len(rs)}
        for m in metrics:
            row[f"{m}_mean"] = sum(r[m] for r in rs) / len(rs)
        new_rows[(method, rate)] = row
    all_rows = list(existing.values()) + list(new_rows.values())
    all_rows.sort(key=lambda r: (r["method"], r["rate"]))
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {summary_path} ({len(all_rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat3dgs_root", default="/work/u4546465/evaluation/CAT3DGSPro_organized")
    ap.add_argument("--hac_root", default="/work/u4546465/evaluation/HACplus+CR")
    ap.add_argument("--out_dir", default="/home/u4546465/iqa_eval/results")
    ap.add_argument("--metrics", nargs="+", default=NR_METRICS, choices=NR_METRICS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_scenes", default=None, help="comma-separated scene names, for a quick test run")
    ap.add_argument("--limit_per_scene", type=int, default=None, help="cap views per scene, for a quick test run")
    ap.add_argument("--methods", default=None,
                    help="comma-separated method names to (re)run, e.g. 'HAC_ori'. "
                         "Default: all (3DCR, Orig, plus all EXTERNAL_METHODS). "
                         "Rows for other methods already in the output CSVs are preserved.")
    ap.add_argument("--no_merge", action="store_true",
                    help="overwrite output CSVs from scratch instead of merging with existing ones")
    args = ap.parse_args()

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views...")
    scenes, _ = discover_views(args.cat3dgs_root, limit_scenes, args.limit_per_scene)
    n_views = sum(len(v) for v in scenes.values())
    print(f"Found {len(scenes)} scenes, {n_views} views total: "
          f"{ {s: len(v) for s, v in scenes.items()} }")

    print(f"Building file records (methods={methods or 'ALL'})...")
    records = build_records(args.cat3dgs_root, args.hac_root, scenes, methods=methods)
    print(f"Total (method,rate,view) records: {len(records)}")

    print(f"Computing NR metrics {args.metrics} on device={args.device} ...")
    records = run_nr_metrics(records, args.metrics, args.device)

    write_csvs(records, args.out_dir, args.metrics, merge=not args.no_merge)


if __name__ == "__main__":
    main()
