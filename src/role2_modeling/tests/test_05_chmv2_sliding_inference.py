"""Tests for 05_chmv2_sliding_inference.py

Run from the repo root:
    pytest src/role2_modeling/tests -v

* Core tests need only numpy and run anywhere (no GPU, no model download).
* GeoTIFF tests need rasterio (skipped if it is not installed).
* The real-model smoke test runs only with CHMV2_SMOKE=1 (needs torch,
  transformers, and `huggingface-cli login` with the gated licence accepted).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "05_chmv2_sliding_inference.py"
_spec = importlib.util.spec_from_file_location("chmv2_inference", SCRIPT)
m = importlib.util.module_from_spec(_spec)
sys.modules["chmv2_inference"] = m
_spec.loader.exec_module(m)

CHIP = 64  # small chips keep the tests fast; the logic is identical for 512


class CountingModel:
    """Wraps a model and counts how many chips it received."""

    def __init__(self, inner):
        self.inner, self.calls, self.chips = inner, 0, 0

    def __call__(self, chips):
        self.calls += 1
        self.chips += len(chips)
        return self.inner(chips)


class EdgeArtifactModel:
    """Correct brightness 'heights' plus a +10 m error in the outer 4 px of every
    chip, mimicking the edge artefacts CNN/ViT decoders produce at tile borders."""

    def __init__(self, width=4, error=10.0):
        self.width, self.error = width, error

    def __call__(self, chips):
        out = m.DummyBrightnessModel()(chips)
        w = self.width
        out[:, :w, :] += self.error
        out[:, -w:, :] += self.error
        out[:, :, :w] += self.error
        out[:, :, -w:] += self.error
        return out


def expected_brightness(rgb):
    return rgb.astype(np.float32).mean(axis=-1) / 255.0 * 30.0


def random_scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------- core
@pytest.mark.parametrize("length", [1, 63, 64, 65, 200, 1000])
def test_chip_starts_cover_everything(length):
    starts = m.chip_starts(length, CHIP, CHIP // 2)
    covered = np.zeros(max(length, CHIP), bool)
    for s in starts:
        covered[s:s + CHIP] = True
    assert covered[:length].all()
    assert starts[0] == 0
    assert all(b > a for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("shape", [(64, 64), (100, 37), (300, 257), (10, 10), (1, 500)])
def test_constant_image_gives_constant_output_no_seams(shape):
    rgb = np.full(shape + (3,), 128, np.uint8)
    valid = np.ones(shape, bool)
    out, _ = m.sliding_inference(rgb, valid, m.DummyBrightnessModel(), chip=CHIP, overlap=0.5)
    assert out.shape == shape
    np.testing.assert_allclose(out, 128 / 255 * 30, atol=1e-4)


@pytest.mark.parametrize("shape", [(200, 300), (129, 77)])
@pytest.mark.parametrize("window", ["hann", "gaussian", "uniform"])
def test_per_pixel_model_reproduced_exactly(shape, window):
    """A per-pixel model must come back pixel-for-pixel: proves there is no
    offset, flip or crop error anywhere in padding, chipping or stitching."""
    rgb = random_scene(*shape)
    valid = np.ones(shape, bool)
    out, _ = m.sliding_inference(rgb, valid, m.DummyBrightnessModel(), chip=CHIP, window=window)
    np.testing.assert_allclose(out, expected_brightness(rgb), atol=1e-3)


def test_hann_blending_suppresses_chip_edge_artifacts():
    rgb = random_scene(256, 256, seed=1)
    valid = np.ones((256, 256), bool)
    truth = expected_brightness(rgb)
    # 4 px artefact on a 128 px chip (~3 %, comparable to ~16 px on a 512 px chip)
    hann, _ = m.sliding_inference(rgb, valid, EdgeArtifactModel(), chip=128, window="hann")
    flat, _ = m.sliding_inference(rgb, valid, EdgeArtifactModel(), chip=128, window="uniform")
    err_hann = np.abs(hann - truth).max()
    err_flat = np.abs(flat - truth).max()
    assert err_hann < 0.5, err_hann          # +10 m artefact almost fully removed
    assert err_flat > 2.0, err_flat          # without Hann, seams are clearly visible


def test_nodata_is_masked_and_empty_chips_are_skipped():
    rgb = random_scene(256, 256, seed=2)
    valid = np.ones((256, 256), bool)
    valid[:, :160] = False                   # left part = nodata (e.g. outside the tile)
    valid[200:210, 200:210] = False          # a small cloud
    counter = CountingModel(m.DummyBrightnessModel())
    out, stats = m.sliding_inference(rgb, valid, counter, chip=CHIP)
    assert np.isnan(out[~valid]).all()
    assert np.isfinite(out[valid]).all()
    np.testing.assert_allclose(out[valid], expected_brightness(rgb)[valid], atol=1e-3)
    assert stats.n_chips_skipped > 0
    assert counter.chips == stats.n_chips_run < stats.n_chips_total


def test_nodata_values_do_not_leak_into_valid_pixels():
    """Garbage values under the nodata mask must not change valid predictions
    of a per-pixel model."""
    rgb = random_scene(150, 150, seed=3)
    valid = np.ones((150, 150), bool)
    valid[50:100, 50:100] = False
    noisy = rgb.copy()
    noisy[~valid] = 255
    a, _ = m.sliding_inference(rgb, valid, m.DummyBrightnessModel(), chip=CHIP)
    b, _ = m.sliding_inference(noisy, valid, m.DummyBrightnessModel(), chip=CHIP)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[valid], b[valid], atol=1e-5)


def test_batch_size_does_not_change_result():
    rgb = random_scene(180, 140, seed=4)
    valid = np.ones((180, 140), bool)
    a, _ = m.sliding_inference(rgb, valid, EdgeArtifactModel(), chip=CHIP, batch_size=1)
    b, _ = m.sliding_inference(rgb, valid, EdgeArtifactModel(), chip=CHIP, batch_size=7)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_bad_inputs_raise():
    valid = np.ones((64, 64), bool)
    with pytest.raises(ValueError):
        m.sliding_inference(np.zeros((64, 64, 3), np.uint16), valid, m.DummyBrightnessModel(), chip=CHIP)
    with pytest.raises(ValueError):
        m.sliding_inference(np.zeros((64, 64, 3), np.uint8), valid, m.DummyBrightnessModel(), chip=60)
    with pytest.raises(RuntimeError):
        m.sliding_inference(np.zeros((64, 64, 3), np.uint8), valid, lambda c: np.zeros((1, 2, 2)), chip=CHIP)


def test_to_uint8_modes():
    rng = np.random.default_rng(5)
    bands16 = rng.integers(200, 4000, size=(3, 50, 50)).astype(np.uint16)
    valid = np.ones((50, 50), bool)

    rgb, info = m.to_uint8(bands16, valid, "auto")
    assert rgb.shape == (50, 50, 3) and rgb.dtype == np.uint8
    assert info["mode"] == "percentile" and rgb.min() == 0 and rgb.max() == 255

    rgb8, info8 = m.to_uint8(bands16.astype(np.uint8), valid, "auto")
    assert info8["mode"] == "none"
    np.testing.assert_array_equal(rgb8, np.moveaxis(bands16.astype(np.uint8), 0, -1))

    rgbf, _ = m.to_uint8(bands16, valid, "fixed", fixed_range=(0, 4000))
    np.testing.assert_allclose(rgbf[..., 0], np.round(bands16[0] / 4000 * 255), atol=1)

    with pytest.raises(ValueError):
        m.to_uint8(bands16, valid, "none")


# ------------------------------------------------------------------ GeoTIFF
def _write_tif(path, data, nodata=None, crs="EPSG:32635", res=0.6):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(425000.0, 4925000.0, res, res)  # somewhere over Bucharest
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(data)
    return transform


def _run(tmp_path, *extra):
    return m.main([*extra, "--dummy", "--chip", str(CHIP), "--output-dir", str(tmp_path / "pred")])


def test_geotiff_roundtrip_keeps_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    data = np.moveaxis(random_scene(150, 210, seed=6), -1, 0)
    src = tmp_path / "scene.tif"
    transform = _write_tif(src, data)

    assert _run(tmp_path, str(src)) == 0
    out = tmp_path / "pred" / "scene_dummy.tif"
    with rasterio.open(out) as ds:
        assert ds.crs.to_string() == "EPSG:32635"
        assert ds.transform == transform
        assert (ds.width, ds.height, ds.count) == (210, 150, 1)
        assert ds.dtypes[0] == "float32" and ds.nodata == m.NODATA_OUT
        h = ds.read(1)
        assert "summary" in ds.tags()
    np.testing.assert_allclose(h, expected_brightness(np.moveaxis(data, 0, -1)), atol=1e-3)


def test_geotiff_band_order_window_nodata_and_mask(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    rgb = random_scene(160, 160, seed=7)
    bgr = np.moveaxis(rgb[..., ::-1], -1, 0).copy()          # stored as B,G,R like many VHR products
    bgr[:, :20, :] = 0                                         # black border = nodata
    src = tmp_path / "bgr.tif"
    _write_tif(src, bgr, nodata=0)
    cloud = np.zeros((1, 160, 160), np.uint8)
    cloud[0, 100:120, 100:120] = 1
    mask = tmp_path / "cloud.tif"
    _write_tif(mask, cloud)

    assert _run(tmp_path, str(src), "--bands", "3", "2", "1",
                "--exclude-mask", str(mask), "--window", "0", "0", "130", "140") == 0
    with rasterio.open(tmp_path / "pred" / "bgr_dummy.tif") as ds:
        assert (ds.width, ds.height) == (130, 140)
        h = ds.read(1, masked=True)
    assert h.mask[:20, :].all()                                # nodata border
    assert h.mask[100:120, 100:120].all()                      # cloud
    ok = ~h.mask
    truth = expected_brightness(rgb[:140, :130])
    np.testing.assert_allclose(h.data[ok], truth[ok], atol=1e-3)


def test_wrong_crs_is_rejected(tmp_path):
    pytest.importorskip("rasterio")
    src = tmp_path / "wgs84.tif"
    _write_tif(src, np.moveaxis(random_scene(64, 64), -1, 0), crs="EPSG:4326", res=0.00001)
    assert _run(tmp_path, str(src)) == 1                       # contract violation -> failure
    assert _run(tmp_path, str(src), "--allow-any-crs") == 0


# --------------------------------------------------------------- real model
@pytest.mark.skipif(os.environ.get("CHMV2_SMOKE") != "1", reason="set CHMV2_SMOKE=1 to run the real model")
def test_real_chmv2_smoke():
    model = m.CHMv2Model("facebook/dinov3-vitl16-chmv2-dpt-head")
    chips = random_scene(512, 512, seed=8)[None].repeat(2, axis=0)
    out = model(chips)
    assert out.shape == (2, 512, 512) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert -1.0 <= out.min() and out.max() <= 100.0            # metres, model range ~0-96
    np.testing.assert_allclose(out[0], out[1], atol=0.05)      # deterministic
