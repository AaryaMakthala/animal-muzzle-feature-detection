# Animal Muzzle Region Detection

## Overview

This project uses YOLO-based object detection to localize regions of the animal muzzle with bounding boxes. The model detects four classes: left nostril, right nostril, upper lip, and lower lip. The implementation is a single Jupyter notebook that covers the workflow from dataset preparation to inference.

## Approach

```text
Dataset preparation
        ↓
Annotation processing
        ↓
Class mapping
        ↓
Grouped train/validation split
        ↓
YOLO dataset construction
        ↓
Model training
        ↓
Validation
        ↓
Inference
```

- Dataset preparation: an archive of images and bounding-box labels is extracted, and the `images` and `boxes` directories are located by name.
- Annotation processing: images and label files are paired by filename stem, and unpaired files are excluded.
- Class mapping: the `muzzle` class in the source annotations is dropped, and the remaining classes are remapped to indices 0 to 3.
- Grouped train/validation split: frames are grouped by source video before splitting, so no video contributes frames to both sets. The split is deterministic for a fixed seed (42) and split ratio (0.85).
- YOLO dataset construction: images and remapped labels are written to a YOLO-format directory with a `data.yaml` file.
- Training, validation, and inference are performed with Ultralytics.

## Dataset

The source annotations are CVAT-style bounding boxes exported as YOLO text files. The images are frames extracted from short videos, which is why the split is grouped by source video: consecutive frames are nearly identical, and a random frame-level split would place near-duplicates in both the training and validation sets.

Exact dataset sizes are withheld.

| Statistic | Value |
|---|---|
| Source videos or samples | X |
| Matched image-label pairs | X |
| Training images | X |
| Validation images | X |
| Validation instances | X |

The dataset is not included in this repository.

## Model

- Architecture: `yolov8s` (Ultralytics 8.4.138), initialized from the pretrained `yolov8s.pt` checkpoint with 4 classes.
- Training configuration: image size 640, batch size 16, AdamW optimizer, up to 150 epochs, early stopping with patience 30, horizontal flip probability (`fliplr`) 0.5. Other hyperparameters were left at the Ultralytics defaults.
- Training was performed with GPU acceleration in Google Colab. In the recorded run, early stopping ended training after 76 epochs, and the best epoch was 46.

## Results

Metrics were computed by Ultralytics on the validation split using the best checkpoint from the recorded training run.

| Metric | Value |
|---|---:|
| Precision | 0.872 |
| Recall | 0.870 |
| mAP50 | 0.918 |
| mAP50-95 | 0.621 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| left nostril | 0.807 | 0.775 | 0.859 | 0.554 |
| right nostril | 0.778 | 0.785 | 0.846 | 0.546 |
| upper lip | 0.968 | 0.975 | 0.988 | 0.700 |
| lower lip | 0.935 | 0.943 | 0.981 | 0.682 |

## Usage

1. Install the dependencies: `pip install -r requirements.txt`. In Google Colab, PyTorch is already available, and the notebook installs the pinned Ultralytics version itself.
2. Open `notebooks/muzzle_region_detection_yolov8.ipynb` in Jupyter or Google Colab. A CUDA-capable GPU is needed for training.
3. To reproduce the training, provide your own dataset as a zip archive containing two directories: `images` (`.jpg`, `.jpeg`, or `.png` frames) and `boxes` (one YOLO-format `.txt` label file per image with a matching filename stem, using source class indices 0 left nostril, 1 right nostril, 2 muzzle, 3 upper lip, 4 lower lip).
4. Set `DATASET_ARCHIVE_PATH` and the other paths in the configuration cell, then run the cells in order.
5. For inference, set `NEW_IMAGE_PATH` in the final cell to an image of your own. The notebook does not include prediction examples.

## Limitations and Scope

- The dataset, annotations, and trained weights are not included, so the training and validation cells cannot be run without user-supplied data.
- The reported results come from a single recorded validation run. The same validation split was used for early stopping and for selecting the best checkpoint, and no separate held-out test set was evaluated.
- Horizontal flip augmentation was enabled (`fliplr=0.5`) during the recorded run. No separate ablation study was performed to measure its effect on side-specific classes.
