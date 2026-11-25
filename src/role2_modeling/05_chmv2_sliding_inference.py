#!/usr/bin/env python3
"""
05_chmv2_sliding_inference.py  (Role 2 — Foundation Model & ML Lead)

Zero-shot canopy-height inference with Meta/WRI CHMv2
(facebook/dinov3-vitl16-chmv2-dpt-head) over large GeoTIFF cutouts.

How it works
------------
1. Read an RGB GeoTIFF (EPSG:32635) plus its validity mask (nodata / alpha /
   optional external cloud mask from Script 02).
2. Convert to uint8 0-255 (CHMv2's image processor expects 0-255 input).
3. Reflect-pad the scene by half a chip, cut it into 512 x 512 chips with
   50 % overlap, run CHMv2 on batches of chips.
4. Blend overlapping chip predictions with a 2D Hann window (weights ~0 at
   chip edges, 1 at chip centres) and normalise by the summed weights, so
   tile-edge artefacts are suppressed and there are no seams.
5. Write a float32 height raster (metres) on exactly the same grid
   (CRS, transform, width, height) as the input, nodata = -9999.

Usage examples
--------------
  # Quick pipeline check without the model (no GPU / no HF login needed):
  python src/role2_modeling/05_chmv2_sliding_inference.py --dummy \
      data/processed/kompsat_east_2021_0p60m.tif

  # Real run on a small window first (col_off row_off width height):
  python src/role2_modeling/05_chmv2_sliding_inference.py \
      --window 0 0 1024 1024 data/processed/kompsat_east_2021_0p60m.tif

  # Full run on every cutout in a folder, with the Script 02 cloud mask:
  python src/role2_modeling/05_chmv2_sliding_inference.py \
      --exclude-mask data/processed/kompsat_east_2022_cloudmask.tif \
      data/processed/kompsat_east_2022_0p60m_picf.tif

Notes
-----
* Output units: the model config's depth range (0.001-96) and the CHMv2 paper
  indicate metres; verify once against the static WRI map (Script 06).
* Band order: many VHR products store bands as B,G,R(,NIR). Use --bands to
  pick R,G,B explicitly (1-based), e.g. --bands 3 2 1.
* For the 14.2 deg vs 24.5 deg comparison, feed PICF-matched uint8 cutouts
  from Script 04 so that no per-scene stretch is applied here
  (--stretch auto does nothing for uint8 input).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

LOG = logging.getLogger("chmv2_inference")

NODATA_OUT = -9999.0
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paths_and_constants.yaml"

# A chip model takes a batch of uint8 RGB chips (N, H, W, 3) and returns
# float32 heights in metres (N, H, W).
ChipModel = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Core tiling and blending (pure numpy, fully unit-testable without a GPU)
# ---------------------------------------------------------------------------
def chip_starts(length: int, chip: int, stride: int) -> list[int]:
    """Start offsets so chips of size `chip` cover [0, length) completely."""
    if length <= chip:
        return [0]
    starts = list(range(0, length - chip + 1, stride))
    if starts[-1] + chip < length:
        starts.append(length - chip)  # last chip flush with the edge
    return starts


def blend_window(size: int, kind: str = "hann") -> np.ndarray:
    """2D blending weights, strictly positive so every pixel gets weight > 0."""
    if kind == "hann":
        w1 = np.hanning(size + 2)[1:-1]  # drop the exact zeros at both ends
    elif kind == "gaussian":
        x = np.arange(size) - (size - 1) / 2.0
        w1 = np.exp(-0.5 * (x / (size / 4.0)) ** 2)
    elif kind == "uniform":
        w1 = np.ones(size)
    else:
        raise ValueError(f"unknown blend window '{kind}'")
    return np.outer(w1, w1).astype(np.float32)


@dataclass
class InferenceStats:
    n_chips_total: int = 0
    n_chips_run: int = 0
    n_chips_skipped: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def sliding_inference(
    rgb: np.ndarray,
    valid: np.ndarray,
    model: ChipModel,
    chip: int = 512,
    overlap: float = 0.5,
    batch_size: int = 4,
    window: str = "hann",
    progress: bool = False,
) -> tuple[np.ndarray, InferenceStats]:
    """Run `model` over a whole scene with overlapping, blended chips.

    rgb   : (H, W, 3) uint8
    valid : (H, W) bool, False = nodata / cloud / outside AOI
    returns (heights (H, W) float32 with NaN where invalid, stats)
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    if valid.shape != rgb.shape[:2]:
        raise ValueError("valid mask shape does not match image")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if chip % 16:
        raise ValueError("chip size must be a multiple of 16 (DINOv3 patch size)")

    t0 = time.time()
    H, W = valid.shape
    stride = max(1, int(round(chip * (1.0 - overlap))))

    # Fill invalid pixels so the model never sees random nodata values,
    # then reflect-pad by half a chip so scene borders sit in chip interiors.
    img = np.where(valid[..., None], rgb, 0).astype(np.uint8)
    pad = chip // 2
    extra_h = max(0, chip - (H + 2 * pad))
    extra_w = max(0, chip - (W + 2 * pad))
    pads = ((pad, pad + extra_h), (pad, pad + extra_w))
    img_p = np.pad(img, pads + ((0, 0),), mode="reflect" if min(H, W) > 1 else "edge")
    valid_p = np.pad(valid, pads, mode="constant", constant_values=False)
    Hp, Wp = valid_p.shape

    weights = blend_window(chip, window)
    acc = np.zeros((Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Hp, Wp), dtype=np.float32)

    positions = [(y, x) for y in chip_starts(Hp, chip, stride) for x in chip_starts(Wp, chip, stride)]
    stats = InferenceStats(n_chips_total=len(positions))
    todo = [(y, x) for (y, x) in positions if valid_p[y:y + chip, x:x + chip].any()]
    stats.n_chips_skipped = len(positions) - len(todo)

    for b in range(0, len(todo), batch_size):
        batch_pos = todo[b:b + batch_size]
        batch = np.stack([img_p[y:y + chip, x:x + chip] for (y, x) in batch_pos])
        pred = np.asarray(model(batch), dtype=np.float32)
        if pred.shape != (len(batch_pos), chip, chip):
            raise RuntimeError(f"model returned {pred.shape}, expected {(len(batch_pos), chip, chip)}")
        for (y, x), p in zip(batch_pos, pred):
            acc[y:y + chip, x:x + chip] += p * weights
            wsum[y:y + chip, x:x + chip] += weights
        stats.n_chips_run += len(batch_pos)
        if progress:
            LOG.info("  chips %d/%d", stats.n_chips_run, len(todo))

    with np.errstate(invalid="ignore", divide="ignore"):
        out_p = np.where(wsum > 0, acc / wsum, np.nan).astype(np.float32)
    out = out_p[pad:pad + H, pad:pad + W]
    out[~valid] = np.nan
    stats.seconds = time.time() - t0
    return out, stats


def to_uint8(
    bands: np.ndarray,
    valid: np.ndarray,
    mode: str = "auto",
    percentiles: tuple[float, float] = (2.0, 98.0),
    fixed_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """(C, H, W) any dtype -> (H, W, C) uint8 and a dict describing the stretch.

    auto       : uint8 passes through untouched, anything else -> percentile
    none       : input must already be uint8
    percentile : per-band linear stretch between the given percentiles of valid pixels
    fixed      : same linear range (lo, hi) for all bands (use it to apply
                 one identical stretch to two scenes)
    """
    if mode == "auto":
        mode = "none" if bands.dtype == np.uint8 else "percentile"
    info: dict = {"mode": mode}
    if mode == "none":
        if bands.dtype != np.uint8:
            raise ValueError(f"--stretch none needs uint8 input, got {bands.dtype}")
        return np.moveaxis(bands, 0, -1).copy(), info

    data = bands.astype(np.float32)
    out = np.zeros(data.shape, dtype=np.uint8)
    ranges = []
    for i, band in enumerate(data):
        if mode == "percentile":
            vals = band[valid] if valid.any() else band.ravel()
            lo, hi = np.percentile(vals, percentiles)
        elif mode == "fixed":
            if fixed_range is None:
                raise ValueError("--stretch fixed needs --stretch-range LO HI")
            lo, hi = fixed_range
        else:
            raise ValueError(f"unknown stretch mode '{mode}'")
        scale = 255.0 / max(float(hi - lo), 1e-6)
        out[i] = np.clip(np.round((band - lo) * scale), 0, 255).astype(np.uint8)
        ranges.append([float(lo), float(hi)])
    info["ranges"] = ranges
    if mode == "percentile":
        info["percentiles"] = list(percentiles)
    return np.moveaxis(out, 0, -1).copy(), info


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class DummyBrightnessModel:
    """Stand-in for CHMv2: 'height' = mean RGB brightness scaled to 0-30 m.

    It is purely per-pixel, so a correct tiling/blending pipeline must return
    exactly mean(RGB)/255*30 everywhere. Used by --dummy and the tests.
    """

    name = "dummy-brightness"

    def __call__(self, chips: np.ndarray) -> np.ndarray:
        return (chips.astype(np.float32).mean(axis=-1) / 255.0 * 30.0).astype(np.float32)


class CHMv2Model:
    """Thin wrapper around the Hugging Face CHMv2 checkpoint."""

    def __init__(self, hf_id: str, dtype: str = "float16", device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        torch_dtype = getattr(torch, dtype)
        if device == "cpu" and torch_dtype == torch.float16:
            LOG.warning("float16 is slow/unsupported on CPU, using float32")
            torch_dtype = torch.float32
        self.device, self.dtype, self.name = device, torch_dtype, hf_id

        LOG.info("Loading %s on %s (%s)...", hf_id, device, torch_dtype)
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id, dtype=torch_dtype)
        self.model.to(device).eval()

    def __call__(self, chips: np.ndarray) -> np.ndarray:
        torch = self.torch
        n, h, w, _ = chips.shape
        # Chips are already 512 px (multiple of 16) -> no resize, no padding.
        inputs = self.processor(images=list(chips), return_tensors="pt", do_resize=False)
        inputs = {
            k: (v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        outputs.predicted_depth = outputs.predicted_depth.float()
        post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)] * n)
        return np.stack([p["predicted_depth"].cpu().numpy() for p in post]).astype(np.float32)


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def read_scene(path: Path, bands: Sequence[int] | None, window: Sequence[int] | None,
               zero_is_nodata: bool, exclude_mask: Path | None):
    """Return (bands (C,H,W), valid (H,W) bool, georef dict)."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        idx = list(bands) if bands else [1, 2, 3]
        if src.count < max(idx):
            raise ValueError(f"{path.name}: has {src.count} bands, asked for {idx}")
        win = Window(*window) if window else None
        data = src.read(idx, window=win)
        masks = src.read_masks(idx, window=win)
        transform = src.window_transform(win) if win else src.transform
        georef = {"crs": src.crs, "transform": transform, "res": src.res,
                  "width": data.shape[2], "height": data.shape[1]}

    valid = np.all(masks > 0, axis=0)
    if zero_is_nodata:
        valid &= np.any(data != 0, axis=0)

    if exclude_mask is not None:
        with rasterio.open(exclude_mask) as msrc:
            mwin = None
            if window:
                mwin = Window(*window)
            if msrc.crs != georef["crs"] or (
                (msrc.window_transform(mwin) if mwin else msrc.transform) != georef["transform"]
            ):
                raise ValueError(f"{exclude_mask.name} is not on the same grid as {path.name}")
            m = msrc.read(1, window=mwin)
        valid &= m == 0  # non-zero = cloud / shadow / excluded
    return data, valid, georef


def write_heights(path: Path, heights: np.ndarray, georef: dict, tags: dict) -> None:
    import rasterio

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": georef["width"], "height": georef["height"],
        "crs": georef["crs"], "transform": georef["transform"],
        "nodata": NODATA_OUT, "compress": "deflate", "predictor": 3, "BIGTIFF": "IF_SAFER",
    }
    if georef["width"] >= 256 and georef["height"] >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(heights), heights, NODATA_OUT).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "canopy_height_m")
        dst.update_tags(**{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in tags.items()})


def height_summary(h: np.ndarray) -> dict:
    v = h[np.isfinite(h)]
    if v.size == 0:
        return {"valid_px": 0}
    return {
        "valid_px": int(v.size),
        "min": round(float(v.min()), 2),
        "p50": round(float(np.percentile(v, 50)), 2),
        "p95": round(float(np.percentile(v, 95)), 2),
        "max": round(float(v.max()), 2),
        "frac_gt_2m": round(float((v > 2.0).mean()), 4),
    }


def process_file(path: Path, out_dir: Path, model: ChipModel, cfg: dict, args) -> Path | None:
    out_path = out_dir / f"{path.stem}{args.suffix}.tif"
    if out_path.exists() and not args.overwrite:
        LOG.info("skip %s (exists, use --overwrite)", out_path.name)
        return None

    bands, valid, georef = read_scene(path, args.bands, args.window, not args.keep_zeros, args.exclude_mask)

    expected_crs = cfg.get("crs", "EPSG:32635")
    if georef["crs"] is None or georef["crs"].to_string() != expected_crs:
        msg = f"{path.name}: CRS {georef['crs']} != {expected_crs} (shared data contract)"
        if not args.allow_any_crs:
            raise ValueError(msg + "; pass --allow-any-crs to override")
        LOG.warning(msg)

    gsd = float(abs(georef["res"][0]))
    target = float(cfg.get("model", {}).get("target_gsd_m", 0.60))
    LOG.info("%s: %dx%d px, GSD %.2f m (CHMv2 training GSD %.2f m), %.1f%% valid",
             path.name, georef["width"], georef["height"], gsd, target, 100 * valid.mean())
    if abs(gsd - target) / target > 0.05:
        LOG.info("  note: GSD differs from %.2f m -- expected for Exp 2 native-GSD runs", target)

    rgb, stretch_info = to_uint8(bands, valid, args.stretch, tuple(args.percentiles),
                                 tuple(args.stretch_range) if args.stretch_range else None)

    heights, stats = sliding_inference(rgb, valid, model, chip=args.chip, overlap=args.overlap,
                                       batch_size=args.batch_size, window=args.blend, progress=args.verbose)
    summary = height_summary(heights)
    LOG.info("  %d chips run, %d skipped, %.1f s | heights %s",
             stats.n_chips_run, stats.n_chips_skipped, stats.seconds, summary)

    tags = {
        "source_file": path.name,
        "model": getattr(model, "name", type(model).__name__),
        "chip_size_px": args.chip, "chip_overlap": args.overlap, "blend_window": args.blend,
        "bands_rgb": list(args.bands) if args.bands else [1, 2, 3],
        "window": list(args.window) if args.window else "full",
        "stretch": stretch_info,
        "input_gsd_m": gsd,
        "units": "metres",
        "summary": summary,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_heights(out_path, heights, georef, tags)
    LOG.info("  -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="GeoTIFF files or folders of *.tif")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, help="default: paths.predictions from config")
    p.add_argument("--suffix", default="_chmv2", help="appended to the input file stem")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dummy", action="store_true", help="use the brightness dummy model (no GPU/HF needed)")
    p.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    p.add_argument("--dtype", help="float16 / float32 (default: config model.dtype)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chip", type=int, help="chip size px (default: config)")
    p.add_argument("--overlap", type=float, help="chip overlap 0-1 (default: config)")
    p.add_argument("--blend", choices=["hann", "gaussian", "uniform"], help="default: config")
    p.add_argument("--bands", type=int, nargs=3, metavar=("R", "G", "B"), help="1-based band indices")
    p.add_argument("--window", type=int, nargs=4, metavar=("COL", "ROW", "W", "H"),
                   help="only process this pixel window (for quick tests)")
    p.add_argument("--exclude-mask", type=Path, help="raster on same grid, non-zero = exclude (clouds)")
    p.add_argument("--keep-zeros", action="store_true", help="do not treat all-zero pixels as nodata")
    p.add_argument("--stretch", choices=["auto", "none", "percentile", "fixed"], default="auto")
    p.add_argument("--percentiles", type=float, nargs=2, default=[2.0, 98.0])
    p.add_argument("--stretch-range", type=float, nargs=2, metavar=("LO", "HI"))
    p.add_argument("--allow-any-crs", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config) if args.config.exists() else {}
    mcfg = cfg.get("model", {})
    args.chip = args.chip or int(mcfg.get("chip_size_px", 512))
    args.overlap = args.overlap if args.overlap is not None else float(mcfg.get("chip_overlap", 0.5))
    args.blend = args.blend or mcfg.get("blend_window", "hann")
    out_dir = args.output_dir or REPO_ROOT / cfg.get("paths", {}).get("predictions", "data/predictions")

    files: list[Path] = []
    for p in args.inputs:
        files += sorted(p.glob("*.tif")) if p.is_dir() else [p]
    if not files:
        LOG.error("no input GeoTIFFs found")
        return 1

    if args.dummy:
        model: ChipModel = DummyBrightnessModel()
        if args.suffix == "_chmv2":
            args.suffix = "_dummy"
    else:
        model = CHMv2Model(mcfg.get("hf_id", "facebook/dinov3-vitl16-chmv2-dpt-head"),
                           dtype=args.dtype or mcfg.get("dtype", "float16"), device=args.device)

    failed = 0
    for f in files:
        try:
            process_file(f, out_dir, model, cfg, args)
        except Exception as e:  # keep going over the other files
            failed += 1
            LOG.error("%s failed: %s", f.name, e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
