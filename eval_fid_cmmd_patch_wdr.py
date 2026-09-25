#!/usr/bin/env python
"""
HiFiC-patchify FID + CMMD for the cat3dgs_w_wd_r dataset, which has its own
layout distinct from every other dataset handled so far:

  Ground truth (shared, reused from CAT3DGSPro_organized):
    CAT3DGSPro_organized/{scene}_{idx:05d}/reference.png

  Each of 7 independent methods (folder name == method name), with its own
  root dir and a per-method scene-folder-name quirk:
    cat3dgs_w_wd_r/low_rate_size1.35/{scene}/{idx:05d}_{psnr}.png
    cat3dgs_w_wd_r/mid_rate_size3.2/{scene}_lmbda_0.08/{idx:05d}_{psnr}.png   <- note suffix
    cat3dgs_w_wd_r/high_rate_size9.32/{scene}/{idx:05d}_{psnr}.png
    cat3dgs_w_wd_r/wdr_refine_15k_same_psnr_as_catpro/{scene}/{idx:05d}_{psnr}.png
    cat3dgs_w_wd_r/wdr_refinement_15k_highrate/{scene}/{idx:05d}_{psnr}.png
    cat3dgs_w_wd_r/wdr_refinement_15k_lowrate/{scene}/{idx:05d}_{psnr}.png
    cat3dgs_w_wd_r/wdr_refinement_15k_midrate/{scene}/{idx:05d}_{psnr}.png

  The trailing _{psnr} in each filename is per-view PSNR (not needed here) and
  isn't predictable, so files are matched by glob on the zero-padded idx prefix.
  View counts per scene were verified to exactly match CAT3DGSPro_organized's
  (e.g. bicycle=25, bonsai=37, ...), confirming index alignment with the shared
  reference.png set.

No rate axis (each of the 7 folders is treated as one independent method, per
user's explicit choice -- NOT grouped into low/mid/high of a shared method).

Reuses compute_all_patches_and_features / compute_per_scene / write_outputs
from eval_fid_cmmd_patch_tgcompare.py unchanged (same (method, scene, idx)
record shape, no rate field) -- only the discovery/record-building step here
is new.

Usage:
  python eval_fid_cmmd_patch_wdr.py --no_downsample \
      --out_dir /home/u4546465/iqa_eval/results/wdr --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_cat3dgs import discover_views  # noqa: E402
from eval_fid_cmmd_patch_tgcompare import (  # noqa: E402
    compute_all_patches_and_features, compute_per_scene, write_outputs,
)

WDR_ROOT = "/work/u4546465/evaluation/cat3dgs_w_wd_r"

# method_name -> (subdir under WDR_ROOT, scene-folder-name suffix)
METHODS_CONFIG = {
    "low_rate_size1.35": ("low_rate_size1.35", ""),
    "mid_rate_size3.2": ("mid_rate_size3.2", "_lmbda_0.08"),
    "high_rate_size9.32": ("high_rate_size9.32", ""),
    "wdr_refine_15k_same_psnr_as_catpro": ("wdr_refine_15k_same_psnr_as_catpro", ""),
    "wdr_refinement_15k_highrate": ("wdr_refinement_15k_highrate", ""),
    "wdr_refinement_15k_lowrate": ("wdr_refinement_15k_lowrate", ""),
    "wdr_refinement_15k_midrate": ("wdr_refinement_15k_midrate", ""),
}


def build_records(cat3dgs_root, wdr_root, scenes, ref_paths, methods=None):
    records = []
    for method_name, (subdir, scene_suffix) in METHODS_CONFIG.items():
        if methods is not None and method_name not in methods:
            continue
        for scene, idxs in scenes.items():
            for idx in idxs:
                scene_dir = Path(wdr_root) / subdir / f"{scene}{scene_suffix}"
                matches = sorted(scene_dir.glob(f"{idx}_*.png"))
                if not matches:
                    print(f"WARNING: missing {scene_dir}/{idx}_*.png", file=sys.stderr)
                    continue
                records.append(dict(method=method_name, scene=scene, idx=idx,
                                     path=matches[0], ref_path=ref_paths[(scene, idx)]))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat3dgs_root", default="/work/u4546465/evaluation/CAT3DGSPro_organized")
    ap.add_argument("--wdr_root", default=WDR_ROOT)
    ap.add_argument("--out_dir", default="/home/u4546465/iqa_eval/results/wdr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_scenes", default=None)
    ap.add_argument("--limit_per_scene", type=int, default=None)
    ap.add_argument("--methods", default=None, help="comma-separated subset of: " + ",".join(METHODS_CONFIG))
    ap.add_argument("--no_downsample", action="store_true")
    args = ap.parse_args()

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views (ground truth from CAT3DGSPro_organized)...")
    scenes, ref_paths = discover_views(args.cat3dgs_root, limit_scenes, args.limit_per_scene)
    print(f"Found {len(scenes)} scenes, {sum(len(v) for v in scenes.values())} views total")

    print(f"Building file records (methods={methods or 'ALL 7'})...")
    records = build_records(args.cat3dgs_root, args.wdr_root, scenes, ref_paths, methods=methods)
    print(f"Total (method,view) records: {len(records)} (each -> HiFiC two-grid 256x256 patches)")

    print(f"Computing HiFiC-patch Inception (FID) + CLIP (CMMD) features "
          f"(no_downsample={args.no_downsample})...")
    fake_fid, fake_cmmd, ref_fid, ref_cmmd = compute_all_patches_and_features(
        records, ref_paths, args.device, args.no_downsample)

    print("Computing per-scene FID + CMMD...")
    per_scene = compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd)

    prefix = "patch_nodownsample" if args.no_downsample else "patch"
    write_outputs(per_scene, args.out_dir, prefix)


if __name__ == "__main__":
    main()
