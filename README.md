# Same Trees, Different Look — Bucharest CHMv2 Urban Audit

**A ground-truth-free audit of how viewing geometry, illumination and pixel scale affect Meta & WRI's open Canopy Height Model v2 (CHMv2) in cities.**

ROSPIN Satellite Data Processing Masterclass · Scientific Research Track · Bucharest, Romania

---

## Why this project exists

Cities need 1 m canopy height maps to plan green infrastructure, estimate carbon storage and mitigate urban heat islands. Airborne LiDAR is expensive and is not openly available for most Eastern European cities, including Bucharest. Meta & WRI's open **CHMv2** model (DINOv3 ViT-L backbone + DPT decoder, [`facebook/dinov3-vitl16-chmv2-dpt-head`](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head)) is an attractive alternative, but it was **never validated in urban areas** and its authors recommend masking built-up land.

In cities, off-nadir viewing angles make tall trees lean over streets, low sun angles lengthen shadows, and unharmonized pixel sizes distort predictions. The result can be "phantom trees" on roofs and roads.

**Our approach:** run CHMv2 on multi-angle, multi-resolution satellite passes over the same Bucharest trees, and use OpenStreetMap buildings and roads, buffered away from vegetation, as **zero-height negative controls** (H_true = 0 m). This lets us measure viewing-angle parallax, shadow hallucinations and pixel-scale bias **without LiDAR**.

## Research questions

| # | Question | Data | Controlled → Tested |
|---|----------|------|---------------------|
| **RQ1** – Off-nadir geometry | How much does a ~10° jump in off-nadir angle (14.2° → 24.5°) shift tree footprints horizontally and destabilize predicted heights vertically? | KOMPSAT-3A East pair, 2021-07-29 vs. 2022-07-19 | Sensor (0.55 m), season, sun elevation (~63°) → off-nadir angle |
| **RQ2** – Pixel scale (GSD) | CHMv2 expects ~0.60 m GSD. How does feeding native-resolution imagery bias urban tree heights? | Parcul Grozăvești, 661 crowns: Pléiades Neo 0.30 m, KOMPSAT-3A 0.55 m, SPOT 7 1.50 m, static WRI CHMv2 | Same crowns → native vs. resampled 0.60 m GSD |
| **RQ3** – Negative controls | When CHMv2 predicts height on buildings/roads, how much is pure hallucination vs. off-nadir tree spillover? | OSM + Global Building Atlas footprints, VHR / Sentinel-2 NDVI mask | H_true = 0 m → Zone 1 (>15 m from trees) vs. Zone 2 (0–15 m), at 14.2° vs. 24.5° |
| *Stretch* – Solar azimuth | How do noon vs. afternoon shadows change heights and false positives? | Pléiades Neo 2023-08-02 vs. KOMPSAT-3A West 2022-06-22 | Common 0.60 m GSD → shadow direction |

## Key methods

- **Parallax:** a tree of height *h* seen at off-nadir angle θ is displaced by **Δd = h · tan(θ)** (20 m tree: 5.1 m at 14.2°, 9.1 m at 24.5°).
- **Stratified negative controls:** NDVI vegetation mask + 15 m buffer split built-up areas into **Zone 1** (pure built-up, any predicted height is a hallucination) and **Zone 2** (canopy spillover: parallax + decoder edge bleeding).
- **Shift vs. height:** crowns are scored at object level (ΔH_p95, ΔH_max, MAE, RMSE, Bland–Altman), with 2D cross-correlation to measure horizontal crown shift (Δx, Δy) separately from vertical instability (ΔĤ).
- **Pre-processing:** cloud / cloud-shadow masking, sub-pixel co-registration on flat road intersections only, PICF (pseudo-invariant roof) radiometric matching, bicubic resampling to 0.60 m.
- **Inference:** zero-shot CHMv2, 512 × 512 sliding windows, 50% overlap, Hann blending, fp16.

## Repository structure

```
bucharest-chmv2-audit/
├── config/
│   └── paths_and_constants.yaml   # Shared CRS (EPSG:32635), AOIs, 15 m buffer, 512 px chips
├── data/                          # NOT committed (see .gitignore). Only the READMEs are tracked.
│   ├── raw/                       # Original Masterclass tiles + _Aux.xml / .DIM metadata
│   ├── processed/                 # Co-registered, cloud-masked, resampled GeoTIFFs   (Role 1)
│   ├── predictions/               # CHMv2 height rasters, float32 metres              (Role 2)
│   └── vectors/                   # OSM buildings/roads, Zone 1/2 masks, park crowns  (Role 3)
├── src/
│   ├── role1_geospatial/          # 01–03: metadata & overlap, cloud mask & co-registration, resampling
│   ├── role2_modeling/            # 04–06: PICF matching, CHMv2 inference, static WRI map + swipe_demo/
│   └── role3_audit/               # 07–09: negative controls, crown/control scoring, paper figures
├── figures/                       # 300 DPI figures for the paper
├── paper/                         # LaTeX manuscript
├── requirements.txt
└── README.md
```

**Data contract:** every raster and vector layer uses **EPSG:32635** (WGS 84 / UTM 35N).

## Pipeline

| Step | Script | Owner |
|------|--------|-------|
| 01 | `01_parse_metadata_and_overlap.py` – read off-nadir angle, azimuths, sun elevation; build overlap AOIs | Role 1 |
| 02 | `02_cloud_mask_and_coregister.py` – cloud/shadow mask, ground-only co-registration | Role 1 |
| 03 | `03_resample_and_export_cutouts.py` – native and 0.60 m cutouts | Role 1 |
| 04 | `04_picf_radiometric_match.py` – PICF histogram/moment matching | Role 2 |
| 05 | `05_chmv2_sliding_inference.py` – zero-shot CHMv2 inference | Role 2 |
| 06 | `06_fetch_static_wri_chmv2.py` – clip the published WRI CHMv2 map | Role 2 |
| 07 | `07_build_stratified_controls.py` – OSM footprints, NDVI, Zone 1 / Zone 2 | Role 3 |
| 08 | `08_score_crowns_and_controls.py` – crown and negative-control metrics | Role 3 |
| 09 | `09_generate_paper_figures.py` – Figures 2–5 | Role 3 |

## Setup

Requirements: Python 3.10+, an NVIDIA GPU with 8–12 GB VRAM (or Google Colab), 16 GB+ RAM.

```bash
git clone https://github.com/CsavlovicsAlex/bucharest-chmv2-audit.git
cd bucharest-chmv2-audit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CHMv2 is a gated model: accept the licence on Hugging Face first, then
huggingface-cli login
```

Put the Masterclass imagery in `data/raw/` and set paths in `config/paths_and_constants.yaml`.

## Data & licensing

- **Commercial VHR imagery (KOMPSAT-3A, Pléiades Neo, SPOT 7) is provided under the Masterclass licence and must never be committed.** `.gitignore` excludes everything under `data/` plus common raster/metadata formats anywhere in the repo. Run `git status` before every push.
- Open data used: Sentinel-2 L2A (Copernicus), OpenStreetMap (© OpenStreetMap contributors, ODbL), Global Building Atlas, Meta/WRI static CHMv2 (AWS Open Data).
- CHMv2 weights are subject to Meta's licence on Hugging Face.

## Deliverables (due 11 December 2026)

1. Scientific research paper (up to 10 pages)
2. This reproducible Python repository
3. Presentation and 2-minute demo video (14.2° vs. 24.5° swipe map)

## Key references

- Brandt, J., Tolan, J., et al. (2026). *CHMv2: Improvements in Global Canopy Height Mapping using DINOv3.* [arXiv:2603.06382](https://arxiv.org/abs/2603.06382)
- Dong, T., et al. (2025). High-precision airborne LiDAR remains essential for urban forestry. *Int. J. Appl. Earth Obs. Geoinf.*, 143, 104791. [doi:10.1016/j.jag.2025.104791](https://doi.org/10.1016/j.jag.2025.104791)
- Fogel, F., et al. (2025). Open-Canopy: Towards Very High Resolution Forest Monitoring. *CVPR 2025.* [arXiv:2407.09392](https://arxiv.org/abs/2407.09392)
- Moudrý, V., et al. (2026). Spaceborne Canopy Height Products Should Be Complemented With Airborne Laser Scanning Data. *Earth and Space Science*, 13(1). [doi:10.1029/2025EA004544](https://doi.org/10.1029/2025EA004544)
- Tolan, J., et al. (2024). Very high resolution canopy height maps from RGB imagery. *Remote Sensing of Environment*, 300, 113888. [doi:10.1016/j.rse.2023.113888](https://doi.org/10.1016/j.rse.2023.113888)
