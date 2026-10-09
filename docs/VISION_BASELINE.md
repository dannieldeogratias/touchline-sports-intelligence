# Local vision baseline

Measured on 8 October 2026 on the development laptop. These figures verify the local toolchain; they do not measure football accuracy.

## Environment

- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- Python 3.13.14 in `.venv-vision`
- PyTorch 2.14.1 with CUDA 12.6
- TorchVision 0.29.1
- Ultralytics 8.4.174
- Bundled FFmpeg 7.1 from `imageio-ffmpeg` 0.6.0
- Generic pretrained `yolo26n.pt`, 5.3 MB

Large dependencies, caches, weights, datasets and outputs live under the D: project and are excluded from Git.

## Results

| Check | Result |
|---|---:|
| CUDA available | Yes |
| Reported VRAM | 4.0 GB |
| Synthetic 640 px median inference call | 11.14 ms |
| Synthetic inference rate from median | 89.75 calls/s |
| Synthetic inference peak allocated VRAM | 77.6 MB |
| API-to-worker generated-video test | Passed |
| Fine-tuning smoke test | Passed |
| Smoke training duration | 28.32 s |
| Smoke training peak allocated VRAM | 140.5 MB |

The inference benchmark used an empty generated frame and excludes video decoding, upload time and storage. The training smoke test used four generated 320 px training images for one epoch with batch size one. Its metrics are meaningless for football quality; the successful checkpoint only proves dataset parsing, transfer of pretrained weights, mixed precision, forward/backward passes, optimization and checkpoint writing.

## Reproduce

```powershell
$env:YOLO_CONFIG_DIR=(Resolve-Path '.cache')
.\.venv-vision\Scripts\python.exe ml\benchmark_detector.py --runs 10
.\.venv-vision\Scripts\python.exe ml\smoke_train.py
```

The API launches the isolated worker at `workers/vision/run_detection.py`. The worker samples every fifth video frame, uses ByteTrack to maintain IDs and writes time-indexed boxes plus summary statistics under `data/analysis`. The generic COCO checkpoint remains available for comparison.

## Promotion gate

Before a trained checkpoint replaces the generic baseline:

1. Record footage licenses and match identifiers.
2. Split by whole match before training.
3. Freeze a held-out test manifest.
4. Report player and ball precision/recall separately.
5. Compare the trained checkpoint against this pretrained baseline.
6. Measure video decode-to-result speed and peak VRAM on representative 10-minute footage.
7. Preserve the dataset manifest, configuration, best and last checkpoints, dependency versions and failure cases.

## First football fine-tuning pilot

The CC BY 4.0 Football Image Dataset was audited and used for a bounded three-epoch experiment at 640 px, batch size one. Training completed in 260.31 seconds with 265.6 MB peak allocated VRAM and wrote `models/football-detector-pilot.pt`.

Provided validation-split results:

| Metric | Overall | Ball | Player | Referee |
|---|---:|---:|---:|---:|
| Precision | 0.627 | 1.000* | 0.754 | 0.126 |
| Recall | 0.360 | 0.000 | 0.955 | 0.127 |
| mAP50 | 0.335 | 0.000 | 0.921 | 0.084 |
| mAP50-95 | 0.206 | 0.000 | 0.567 | 0.049 |

`*` The ball precision value is not useful because recall is zero: the model returned no accepted ball detections. Three epochs were enough to adapt player detection but not the small-ball or referee classes. The dataset has only 500 ball boxes versus 8,844 player boxes, and the validation domain mixes real broadcasts with PES 21 gameplay.

## 960 px continuation

The 640 px pilot was continued for five epochs at 960 px because the median labelled ball is only about 18 x 17 pixels in the original 1920 x 1080 images and becomes roughly six pixels wide at 640 px. Training completed in 484.91 seconds with 454.5 MB peak allocated VRAM. The best checkpoint was copied to `models/football-detector-pilot.pt`.

Provided validation-split results:

| Metric | Overall | Ball | Player | Referee |
|---|---:|---:|---:|---:|
| Precision | 0.662 | 0.911 | 0.730 | 0.346 |
| Recall | 0.545 | 0.123 | 0.979 | 0.532 |
| mAP50 | 0.597 | 0.392 | 0.962 | 0.435 |
| mAP50-95 | 0.354 | 0.147 | 0.617 | 0.297 |

The higher resolution materially improved all three classes and established that 960 px training fits comfortably on the 4 GB GPU. Ball recall is still only 0.123, so the checkpoint misses most labelled balls. The validation split also mixes broadcast footage with PES 21 gameplay and cannot be proven independent by match. The local application uses this checkpoint to make the end-to-end workflow testable, while labelling its output experimental and requiring analyst review. It is not a production-quality model.

Next, build a licensed match-grouped dataset with more small-ball examples, hard negatives and consistent referee labels. Evaluate on a frozen real-broadcast test split before connecting this checkpoint to product output. Tiling or a dedicated ball detector should be tested if full-frame 960 px training still misses distant balls.

## Dedicated ball detector and tiled inference

The 612-image dataset was converted into a ball-only detection dataset while retaining all 118 negative images. The dedicated detector initialized from the football pilot and trained for five epochs at 1280 px, batch size one. Training completed in 479.19 seconds with 719.7 MB peak allocated VRAM. Its best checkpoint is `models/football-ball-detector.pt`.

| Metric | Ball-only model |
|---|---:|
| Precision | 0.745 |
| Recall | 0.527 |
| mAP50 | 0.619 |
| mAP50-95 | 0.232 |

The worker now divides each sampled frame into four overlapping tiles, runs the dedicated checkpoint at 960 px per tile, suppresses duplicate boxes and selects a candidate using confidence and recent motion. It can predict through at most four missing sampled frames, marks predicted overlays separately and reports `ball_predicted_frames` so observed and recovered positions remain auditable.

On a three-second validation clip generated from a labelled real-broadcast frame, the end-to-end worker found the ball in all six sampled frames, found 18 player tracks, produced the H.264 annotated video and used 182 MB peak allocated VRAM. This controlled repeat-frame check verifies wiring and overlay placement; it is not an independent accuracy result. The validation recall of 0.527 remains the meaningful current quality estimate.

Further improvement requires match-grouped real-broadcast footage with many more distant-ball examples, occlusions, motion blur, hard negatives and consistent labels. Pass and shot candidates should remain review-only until evaluation on that frozen test set meets the product threshold.

## Camera geometry and tactical analysis

The original single full-pitch homography was removed because broadcast cameras pan, zoom and cut while showing only part of the field. The analyst can now create timestamped calibration segments for a full-pitch, left-half, centre-half or right-half view. Each segment maps four visible field intersections to the corresponding pitch region. Player trails use the calibration active at each frame rather than the calibration at the current playhead.

The worker also emits camera-cut timestamps using colour-histogram discontinuities. A calibration expires at a detected cut, and the tactical view withholds positions until the new camera view is calibrated. It reports visible players by team and withholds formation labels when coverage is insufficient.

The old frame-edge shot heuristic has been disabled. Image direction is not pitch direction in a panning broadcast camera. Automatic shot classification will require calibrated pitch coordinates, a goal-directed multi-frame ball trajectory, preceding possession and a frozen evaluation set. This same world-coordinate layer is the prerequisite for formation snapshots, formation-change intervals, third-man runs and build-up phase recognition.
