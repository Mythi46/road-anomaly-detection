# Road Anomaly Detection Dataset Strategy

This note summarizes public datasets that are potentially useful for the
road-surface anomaly detection project. The goal is not to collect every
dataset blindly, but to build a data plan that improves the current
weak points:

- zero false positives on repaired road surfaces;
- better recall for small / distant potholes;
- robust crack and pothole detection under varied weather, lighting,
  camera angle, and road texture;
- hard-negative suppression for manholes, lane markings, joints,
  shadows, wet patches, and repaired surfaces.

## 1. Working Recommendation

Use public datasets as **supporting data**, not as final proof of
operational readiness.

The highest-value path is:

1. Build a clean public-data training pool from road-damage datasets.
2. Build a hard-negative pool from datasets that include repair, patch,
   manhole, lane marking, and background classes.
3. Keep local/field data as the final validation source.
4. Evaluate with business-facing metrics, not only mAP.

Public datasets can expand the model's base visual vocabulary, but they
cannot replace field data from the actual deployment camera and road
conditions.

## 2. Priority Datasets

### Tier A — Download First

These are the most relevant datasets for the current project.

| Dataset | Source | Size / Classes | Why it matters | Suggested use |
|---|---|---|---|---|
| RDD2022 | Figshare / CRDDC 2022 / Geoscience Data Journal | 47,420 images from Japan, India, Czech Republic, Norway, US, China; 55k+ instances; longitudinal crack, transverse crack, alligator crack, pothole | Best general-purpose road damage base dataset; includes Japan | Pretraining / fine-tuning base |
| RDD2020 | Mendeley Data / RoadDamageDetector project | 26,336 images from India, Japan, Czech Republic; 31k+ instances; D00/D10/D20/D40 | Earlier but still highly relevant; smartphone/vehicle-mounted perspective | Base training and comparison |
| N-RDD2024 | Mendeley Data | 10 defect classes: D00, D10, D20, D30 repaired cracks, D40 potholes, D50 pedestrian crossing blur, D60 lane line blur, D70 manhole, D80 patchy road, D90 rutting | Very useful because it explicitly includes repair-like and hard-negative classes | Hard-negative and repair-aware training |
| PaveDistress | Mendeley Data / related article | High-resolution road images; cracks, patches, potholes, background images without defects | Contains patches and background images; valuable for zero-FP objective | Hard negatives + high-res domain test |
| SVRDD | Scientific Data | 8,000 street-view images, 20,804 instances; cracks, potholes, patches, manhole covers | Useful for street-view/wide-view domain adaptation and manhole/patch hard negatives | External generalization test |
| Road Damage Dataset: Potholes, Cracks, and Manholes | Scientific Reports / Kaggle / Zenodo | 2,009 labeled images; 1,261 potholes, 2,519 cracks, 957 maintenance holes | Recent real-world dataset with maintenance holes; useful for manhole confusion | Hard-negative evaluation and extra training |
| Attain | Mendeley Data | 2,293 images, 19,761 distress instances; cracks, alligator cracks, block cracks, patching/utility cuts, manholes, faded markings, potholes | Strong fit for hard negatives and road marking confusion | Hard-negative mining |
| PaveTrack | Scientific Data / Science Data Bank | 51,012 images for pavement distress identification and 8,928 images for long-term tracking; YOLO object-detection labels | Large-scale, recent, and useful for future cross-frame / cross-date damage tracking | Second base-training pool and tracking study |

### Tier B — Useful Second Wave

These are useful after the Tier A data is downloaded and converted.

| Dataset | Source | Size / Classes | Why it matters | Suggested use |
|---|---|---|---|---|
| Multi-Weather Pothole Detection (MWPD) | Mendeley Data | Pothole boxes under weather variation; small/partially obscured potholes | Directly targets small and weather-affected potholes | Recall improvement for hard potholes |
| Water-filled and Dry Potholes Dataset | Mendeley Data | Images + Pascal VOC / YOLO annotations; includes dashcam videos | Good for puddle/wet pothole ambiguity | Pothole robustness, wet-condition testing |
| RoadDamageVision | Mendeley Data | Drone images from China and Spain; D00/D10/D20/D40/Repair/Block Crack; 7,647 instances | Includes repair class, but aerial viewpoint differs from dashcam | Repair examples; do not over-weight |
| Pavement Distress Dataset | Mendeley Data | Asphalt pavement 6,200 images; complex background subset 212; concrete pavement 4,344 | Useful for background and complex-negative cases | External robustness / hard negatives |
| Cracks and Potholes in Road Images | Mendeley Data / GitHub | 2,235 Brazil highway images; road/crack/pothole masks | Segmentation-style labels for crack/pothole shapes | Optional segmentation-derived boxes |
| GAPs | TU Ilmenau | High-resolution German asphalt pavement distress data; pixel-level labels | Good for top-down distress morphology, manholes/patch-like surfaces | Optional segmentation/domain study |
| StreetSurfaceVis | Scientific Data / Zenodo | 9,122 street-level images, surface type and quality labels | Not object detection for potholes, but useful for surface condition/domain classification | Auxiliary road-surface quality classifier |
| CNRDD | MDPI paper / Roboflow mirror | Eight road damage categories with severity levels | Fine-grained severity and Chinese road conditions | Optional severity/domain study |

### Tier C — Optional / Use Carefully

These can help, but should not be primary sources without inspection.

| Dataset | Source | Notes |
|---|---|---|
| Roboflow Pothole Object Detection | Roboflow Public | Easy YOLO export, but provenance and labels should be inspected |
| Kaggle Potholes Detection YOLOv8 | Kaggle | Useful for quick pothole bootstrapping, but license/provenance needs review |
| Road crack detection Roboflow datasets | Roboflow Universe | Useful for crack diversity, but label quality varies |
| RSD-BD Bangladesh road damage | Mendeley Data | Region-specific road-surface damage; inspect labels before use |
| Asphalt Damage Dataset / ARSDD | Paper references | Promising high-resolution data, but access and license need confirmation |

## 3. Class Mapping

Initial unified YOLO classes should remain small and conservative.

| External label | Internal mapping | Notes |
|---|---|---|
| D00 Longitudinal Crack | `crack` | Could remain subclass later |
| D10 Transverse Crack | `crack` | Could remain subclass later |
| D20 Alligator Crack | `crack` or `surface_deterioration` | For v0, map to `crack` if detection matters |
| D40 Pothole | `pothole` | Primary positive class |
| Repaired crack / repair / patch / patchy road | `repair_negative` | Critical for zero-FP behavior |
| Manhole / maintenance hole | `neutral_negative` | Hard negative, not abnormal |
| Lane marking blur / pedestrian crossing blur / faded markings | `neutral_negative` | Hard negative |
| Rutting / raveling / weathering / block crack | `deferred_damage` | Keep for future; do not force into v0 unless clear |
| Background / no defect | `normal_negative` | Useful for false-positive suppression |

For v0 training, two strategies are possible:

1. **Binary-ish detector**:
   - `pothole`
   - `crack`
   - repair / neutral classes used as empty negative images or ignored boxes.

2. **Multi-bucket detector**:
   - `pothole`
   - `crack`
   - `repair`
   - `neutral`

The current project direction favors the multi-bucket idea because the
decision layer can explicitly compare anomaly evidence and repair
evidence.

## 4. Download / Conversion Order

Recommended order:

1. RDD2022
2. N-RDD2024
3. PaveDistress
4. SVRDD
5. Road Damage Dataset: Potholes, Cracks, and Manholes
6. Attain
7. PaveTrack
8. MWPD
9. Water-filled and Dry Potholes

After those, inspect whether the model still lacks:

- small potholes;
- wet/dark potholes;
- repaired-road negatives;
- manhole negatives;
- road marking / crosswalk negatives;
- Japanese road-surface texture.

Only then add Tier B / Tier C datasets.

## 5. Evaluation Protocol

Do not evaluate only with mAP. The project needs a precision-first
evaluation.

Minimum report per experiment:

| Metric | Why |
|---|---|
| Repaired-road FP count | Main gate |
| Hard-negative FP count | Detects manhole/marking/joint/shadow failure |
| Clearly-visible pothole recall | Demo-facing positive capability |
| Small/distant pothole recall | Known weak point |
| Crack recall | v0 target class |
| Precision at selected operating point | Business-facing trust metric |
| Inference time per frame | Streaming/server feasibility |
| Confusion by data source | Shows domain mismatch |

Suggested validation split:

- **Internal public-data validation**: held-out public images.
- **Field-data validation**: actual PoC / deployment-like images.
- **Hard-negative validation**: repaired surfaces, manholes, markings,
  shadows, wet patches.
- **Streaming validation**: extracted frames from representative video.

## 6. Dataset-Specific Risks

| Risk | Explanation | Mitigation |
|---|---|---|
| Domain mismatch | Public datasets may not match deployment camera, road texture, weather, or resolution | Keep field-data holdout separate |
| Repair under-representation | Many datasets focus on damage, not repaired normal surfaces | Prioritize N-RDD2024, PaveDistress, Attain, and field negatives |
| Label mismatch | Crack categories differ across datasets | Map to small v0 taxonomy first |
| Annotation quality | Roboflow/Kaggle datasets vary widely | Inspect samples before training |
| Viewpoint mismatch | Drone/top-down/street-view may differ from dashcam | Use as auxiliary data, not final validation |
| Over-training public data | Model may become good at public benchmarks but worse on field data | Always select operating point on field/hard-negative validation |

## 7. Immediate Next Steps

1. Create `datasets/catalog.yaml` with dataset name, URL, license, local
   path, classes, annotation format, and intended usage.
2. Download RDD2022 and N-RDD2024 first.
3. Write conversion scripts:
   - `scripts/convert_rdd.py`
   - `scripts/convert_nrdd2024.py`
   - `scripts/convert_pavedistress.py`
4. Build `data/public_yolo/` with unified class mapping.
5. Build `data/hard_negative/` separately.
6. Run a baseline training pass with public data only.
7. Evaluate on the local PoC / hard-negative set without mixing it into
   training.

## 8. Source Links

- RDD2022 Figshare: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547
- RDD2022 paper / arXiv: https://arxiv.org/abs/2209.08538
- RDD2020 Mendeley: https://data.mendeley.com/datasets/5ty2wb6gvg/1
- RoadDamageDetector project: https://github.com/sekilab/RoadDamageDetector
- N-RDD2024 Mendeley: https://data.mendeley.com/datasets/27c8pwsd6v/3
- SVRDD Scientific Data: https://www.nature.com/articles/s41597-024-03263-7
- PaveDistress Mendeley: https://data.mendeley.com/datasets/f46zt2g83x
- Road Damage Dataset: Potholes, Cracks, and Manholes: https://www.nature.com/articles/s41598-026-46679-4
- RoadDamageVision Mendeley: https://data.mendeley.com/datasets/ypm4h4z25c
- MWPD Mendeley: https://data.mendeley.com/datasets/s5hx9n2jc3
- Water-filled and Dry Potholes Mendeley: https://data.mendeley.com/datasets/tp95cdvgm8
- Attain Mendeley: https://data.mendeley.com/datasets/nykrzdm74f
- Pavement Distress Dataset Mendeley: https://data.mendeley.com/datasets/cbm6dkvggn
- Cracks and Potholes in Road Images Mendeley: https://data.mendeley.com/datasets/t576ydh9v8/4
- GAPs: https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/german-asphalt-pavement-distress-dataset-gaps
- StreetSurfaceVis Scientific Data: https://www.nature.com/articles/s41597-024-04295-9
- StreetSurfaceVis Zenodo: https://zenodo.org/records/11449977
- CNRDD paper: https://www.mdpi.com/2076-3417/12/15/7594
- PaveTrack Scientific Data: https://www.nature.com/articles/s41597-025-05748-5
- PaveTrack data DOI: https://doi.org/10.57760/sciencedb.20383
