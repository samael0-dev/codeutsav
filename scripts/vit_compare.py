#!/usr/bin/env python3
"""Compare classical vs ViT (MobileSAM) segmentation: dimension error and per-call latency.

Two modes, both printing real measured numbers only:

* default (synthetic): round props from ``data/ground_truth.csv`` rendered end-on with
  known diameters (``billetvision.synthetic``), calibrated from the marker in each
  frame.  Error is measured against the caliper diameter.
* ``--video PATH``: frames sampled from a real video (e.g. ``data/raw/demo.mp4``).
  There is no caliper truth for those, so the report shows classical-vs-ViT
  *agreement* on width/length, not accuracy.

If torch / mobile_sam / the weights are missing the ViT columns say so and the
script exits with code 2 - it never invents ViT numbers.

    python scripts/vit_compare.py [--video data/raw/demo.mp4] [--frames 12] [--repeats 3]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from billetvision import synthetic  # noqa: E402
from billetvision.vision import vit_segment as V  # noqa: E402
from billetvision.vision.calibrate import calibrate_from_marker  # noqa: E402
from billetvision.vision.measure import measure  # noqa: E402
from billetvision.vision.segment import segment  # noqa: E402

logger = logging.getLogger("vit_compare")

TARGET_PCT = 1.0
TARGET_BILLET_S = 2.0


def _stats(ms: List[float]) -> str:
    return f"mean {np.mean(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms (n={len(ms)})" if ms else "n/a"


def _vit_contour(fn: V.SegmentFn, frame: np.ndarray, box: Optional[tuple], ms: List[float]):
    t0 = time.perf_counter()
    mask = fn(frame, box)
    ms.append((time.perf_counter() - t0) * 1000)
    return V.largest_contour(mask) if mask is not None else None


def run_synthetic(fn: Optional[V.SegmentFn], repeats: int) -> dict:
    rows, cls_ms, vit_ms = [], [], []
    roi = synthetic.ROI_BOX
    for p in (p for p in synthetic.load_ground_truth() if p.shape == "round"):
        for seed in range(repeats):
            frame = synthetic.render_end_view(p.diameter_mm, seed=seed)
            mm = calibrate_from_marker(frame).mm_per_pixel
            t0 = time.perf_counter()
            seg = segment(frame, roi=list(roi), preprocess_kwargs={"auto_exposure": True})
            cls_ms.append((time.perf_counter() - t0) * 1000)
            row = {"prop": p.prop_id, "truth": p.diameter_mm, "classical": None, "vit": None}
            if seg is not None:
                row["classical"] = measure(seg.contour, mm, shape="round", travel_axis="x").diameter_mm
            if fn is not None:
                c = _vit_contour(fn, frame, tuple(roi), vit_ms)
                if c is not None:
                    row["vit"] = measure(c, mm, shape="round", travel_axis="x").diameter_mm
            rows.append(row)
    return {"rows": rows, "classical_ms": cls_ms, "vit_ms": vit_ms[1:] if len(vit_ms) > 1 else vit_ms}


def _err(value: Optional[float], truth: float) -> Optional[float]:
    return None if value is None else (value - truth) / truth * 100.0


def report_synthetic(res: dict, vit_ok: bool, max_calls: int) -> None:
    print("\n## Synthetic round props (caliper truth)\n")
    print("| prop | truth mm | classical mm (err %) | ViT mm (err %) |\n|---|---|---|---|")
    worst = {"classical": 0.0, "vit": 0.0}
    for r in res["rows"]:
        cells = []
        for key in ("classical", "vit"):
            e = _err(r[key], r["truth"])
            if e is not None:
                worst[key] = max(worst[key], abs(e))
            cells.append("n/a" if r[key] is None else f"{r[key]:.2f} ({e:+.2f})")
        print(f"| {r['prop']} | {r['truth']:.1f} | {cells[0]} | {cells[1] if vit_ok else 'unavailable'} |")
    print(f"\nWorst |error|: classical {worst['classical']:.2f}%"
          + (f", ViT {worst['vit']:.2f}%" if vit_ok else ", ViT unavailable"))
    verdict(f"classical max error <= {TARGET_PCT}%", worst["classical"] <= TARGET_PCT)
    if vit_ok:
        verdict(f"ViT max error <= {TARGET_PCT}%", worst["vit"] <= TARGET_PCT)
    report_latency(res, vit_ok, max_calls)


def run_video(path: str, fn: Optional[V.SegmentFn], frames: int, mm_per_px: float) -> dict:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, cls_ms, vit_ms = [], [], []
    for idx in np.linspace(0, max(total - 1, 0), frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        seg = segment(frame, preprocess_kwargs={"auto_exposure": True})
        cls_ms.append((time.perf_counter() - t0) * 1000)
        if seg is None:
            continue
        m_c = measure(seg.contour, mm_per_px, travel_axis="x")
        row = {"frame": int(idx), "classical": (m_c.length_mm, m_c.width_mm), "vit": None}
        if fn is not None:
            x, y, w, h = seg.bounding_rect
            c = _vit_contour(fn, frame, (x, y, x + w, y + h), vit_ms)
            if c is not None:
                m_v = measure(c, mm_per_px, travel_axis="x")
                row["vit"] = (m_v.length_mm, m_v.width_mm)
        rows.append(row)
    cap.release()
    return {"rows": rows, "classical_ms": cls_ms, "vit_ms": vit_ms[1:] if len(vit_ms) > 1 else vit_ms}


def report_video(res: dict, vit_ok: bool, max_calls: int, mm_per_px: float) -> None:
    print(f"\n## Video frames (no caliper truth - agreement only; scale {mm_per_px} mm/px)\n")
    print("| frame | classical L x W mm | ViT L x W mm | width diff % |\n|---|---|---|---|")
    diffs = []
    for r in res["rows"]:
        c, v = r["classical"], r["vit"]
        d = None if v is None else (v[1] - c[1]) / c[1] * 100.0
        if d is not None:
            diffs.append(abs(d))
        print(f"| {r['frame']} | {c[0]:.1f} x {c[1]:.1f} | "
              f"{'unavailable' if not vit_ok else 'n/a' if v is None else f'{v[0]:.1f} x {v[1]:.1f}'} | "
              f"{'' if d is None else f'{d:+.2f}'} |")
    if diffs:
        print(f"\nMax |width difference| classical vs ViT: {max(diffs):.2f}%")
    report_latency(res, vit_ok, max_calls)


def report_latency(res: dict, vit_ok: bool, max_calls: int) -> None:
    print("\n## Latency\n")
    print(f"- classical segment(): {_stats(res['classical_ms'])}")
    if not vit_ok:
        print("- ViT per call: unavailable")
        return
    print(f"- ViT per call (first/warm-up call excluded): {_stats(res['vit_ms'])}")
    if res["vit_ms"]:
        worst_billet_s = max_calls * float(np.percentile(res["vit_ms"], 95)) / 1000.0
        print(f"- worst case per repaired billet: {max_calls} calls x p95 = {worst_billet_s:.2f} s "
              "(runs on the finalizer thread, not the 15 FPS vision loop)")
        verdict(f"ViT repair stays under the {TARGET_BILLET_S:.0f} s per-billet budget", worst_billet_s < TARGET_BILLET_S)


def verdict(label: str, ok: bool) -> None:
    print(f"- {'MEETS' if ok else 'MISSES'} target: {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="compare on frames sampled from this video instead of synthetic props")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled from --video")
    ap.add_argument("--repeats", type=int, default=3, help="noise seeds per round prop (synthetic mode)")
    ap.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg_all = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = V.VitConfig.from_dict({**cfg_all.get("vit_fallback", {}), "enabled": True})
    weights = Path(cfg.weights_path)
    weights = weights if weights.is_absolute() else ROOT / weights
    cfg.weights_path = str(weights)

    fn: Optional[V.SegmentFn] = V.make_segment_fn(cfg)
    probe = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.rectangle(probe, (16, 16), (48, 48), (200, 200, 200), -1)
    fn(probe, (8, 8, 56, 56))                      # warm-up; also tells us whether the ViT can run at all
    vit_ok = V._load_predictor(cfg.model, cfg.weights_path, cfg.device) is not None
    if not vit_ok:
        fn = None
        print("ViT UNAVAILABLE: install requirements-vit.txt and put mobile_sam.pt at "
              f"{cfg.weights_path}. Showing classical numbers only; no ViT numbers are reported.")

    print("# Classical vs ViT segmentation")
    if args.video:
        calib = ROOT / "config/calibration.json"
        mm = 0.5
        if calib.exists():
            import json
            mm = float(json.loads(calib.read_text()).get("mm_per_pixel", mm))
        report_video(run_video(args.video, fn, args.frames, mm), vit_ok, cfg.max_calls_per_billet, mm)
    else:
        report_synthetic(run_synthetic(fn, args.repeats), vit_ok, cfg.max_calls_per_billet)
    return 0 if vit_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
