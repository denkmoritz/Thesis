# Data (not in git)

Inside the repository (relative paths, working directory = repo root):

| Path | Content | Size |
|---|---|---|
| `data/output/stereo_calibration.npz` | saved DVS–RGB stereo calibration (used everywhere) | small |
| `data/output/` other files | calibration v2, zone_params.json / depth_stats.json of the first pipeline | 49 MB |
| `data/ictrap-2022-calibration/` | calibration board recording (dvs.mp4, rgb.mp4) | 579 MB |

Outside the repository, set in `config/paths.yaml`:

| Key | Content |
|---|---|
| `training_root` | `data/` videos `{scene}_dvs.mp4`, `{scene}_rgb.mp4` and DVS annotation CSVs; `dataset/` RGB labels of scenes 1–5; `annotations/6_h-h-h_annotations/` RGB labels of scene 6; built datasets `dataset_dvs`, `dataset_m3`, `dataset_m3d`, `dataset_zone_v2` |
| `raw_data_dir` | `{scene}.h5` event files (trigger timestamps) and scene-6 privacy masks |

Source: recordings and DVS annotations of Gebauer et al. (University of Münster);
RGB labels made for this thesis (CVAT).
