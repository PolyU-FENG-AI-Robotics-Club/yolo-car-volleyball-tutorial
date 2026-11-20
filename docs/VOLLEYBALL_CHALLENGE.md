# Independent assignment: detect a volleyball in a video

## Goal

After completing the guided car tutorial, implement a **volleyball detector** independently. Load your trained checkpoint and the supplied volleyball video in the visual validation interface. Display a bounding box, the label `volleyball`, and a confidence score for the ball, and export the annotated video and prediction records.

This assignment targets the ball, not the players, court, or net. The guided car application remains the starting example; it is not a completed volleyball solution. No training or implementation answer is supplied for this task.

## Training data

Use primaryws's **volleyball_ball_object_detection_dataset, version 1 (rgb_images)** on Roboflow Universe:

- [Dataset and class description](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset)
- [Version 1 and download](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset/dataset/1)

The publisher lists 548 labeled images: 521 training images and 27 validation images, with no test split. The single source category is `ball`. Version 1 contains RGB images resized to 640×640; version 2 uses grayscale. The displayed license is CC BY 4.0. These publisher records were checked on 20 November 2026; the training dataset has not been downloaded or trained in the starter package.

1. Open version 1, select **Download**, and sign in if requested.
2. Choose a YOLOv11 detection ZIP (a compatible YOLOv8 detection export also works).
3. Extract it to `data/volleyball-training/`. Keep the original attribution files.
4. Inspect `data.yaml`, the actual split counts, and at least ten image/label pairs. Read the exported class order rather than assuming it matches COCO.
5. Use only the training split for optimization. Keep the 27 provided validation images outside training; they do not replace the required video demonstration.
6. Check that the same recording as the supplied evaluation video is absent from the training split. Exclude any matching recording before training, and record the revised image count. Different download sources alone do not prove that their images are independent.

Create a one-class target definition, `names: {0: volleyball}`. Map the source `ball` class to ID 0 and preserve its normalized box coordinates. Check that every target label uses ID 0. Keep empty label files for images without a visible ball. This label represents the volleyballs in this exercise; it does not establish that the model can distinguish every other type of sports ball.

## Required validation video

The supplied file is:

`data/validation/volleyball/volleyball-720p.mp4`

It contains **387 decoded frames at approximately 23.976 FPS**, lasting **16.14 seconds**, at 1280×720. It shows a real Paris Volley–Chaumont match. The source is [Shev123's Wikimedia Commons recording](https://commons.wikimedia.org/wiki/File:Paris_Volley_-_Chaumont,_18_February_2014_-_26.webm), released under CC0 1.0. The distributed MP4 is resized and encoded at a fixed frame rate without audio; its timing differs slightly from the original container. Provenance, processing details, and SHA-256 are in `docs/volleyball-video-source.json`.

The complete **v1.1.0 student kit** includes this video. If you use a source-only download, obtain **volleyball-validation-video.zip** from [Releases](https://github.com/PolyU-FENG-AI-Robotics-Club/yolo-car-volleyball-tutorial/releases/tag/v1.1.0) and extract it inside the project folder. It creates `data/validation/volleyball/` automatically. Open the MP4 in a video player before starting the assignment.

This video is the required held-out input for visual validation. **Never train on this video, its extracted frames, crops, or augmentations.** Inspect training-data provenance for overlap before training. Decide training settings before inspecting the final model's results on this video.

## Independent implementation tasks

### 1. Understand the class mismatch

Inspect `model.names` and the car-only filtering in `detect.py`. The supplied COCO model has `sports ball` at ID 32; it has no category named `volleyball`. The custom model for this task uses `volleyball` at ID 0. Resolve categories from the loaded model instead of retaining car ID 2 or blindly reusing ID 32.

As an optional baseline, adapt the class filter to inspect the pretrained model's `sports ball` predictions. Keep that label and report it as a generic-ball baseline. Simply renaming all generic sports-ball predictions to volleyball is not the required training solution.

### 2. Train your volleyball model

Adapt `train_demo.py` to your single-class YAML and training split. Remove the COCO-specific configuration and `classes=[2]`. Retain the provided training-only behavior: no validation loader and no automatic final validation using training images. Setting `val=False` alone is insufficient in the pinned Ultralytics release.

Begin with a one-epoch execution check. Then choose and record a training run, for example 10 epochs, image size 640, batch size 4, CPU, and zero workers. These are starting settings, not an accuracy guarantee. Small balls make input resolution relevant; document any resource-driven changes. Save `last.pt`, the configuration, dataset manifest, and training-loss log. Write the training code and label checks yourself.

### 3. Adapt the visual validation interface

Modify the inference path, interface, and exports so that:

- The interface accepts the custom checkpoint and the supplied volleyball MP4.
- The visible ball receives a readable `volleyball` label, confidence, and bounding box.
- The interface reports ball detections in the current frame; it no longer calls them cars.
- CSV, JSONL, summaries, and validation-report metadata use the actual class name and actual training/video sources. Do not retain hard-coded COCO128, car, or highway-video descriptions for this task.
- Video metadata comes from the selected file. Process 387 frames for this assignment, rather than reusing the car video's 538-frame check.
- Start, Pause/Resume, Stop and save, and video export remain usable. Partial runs are marked incomplete.

The relevant starting files are `detect.py`, `train_demo.py`, `validation_ui.py`, and `validate_video.py`. Keep the original car workflow available, for example through an explicit target mode. A single-class volleyball checkpoint is not expected to detect cars.

### 4. Run and review the volleyball video

After completing your code changes, open the GUI, browse to `data/validation/volleyball/volleyball-720p.mp4`, select your `last.pt`, choose a threshold before the run, and click **Start validation**. Save the results in a separate directory such as `outputs/volleyball-validation/`.

The following command shows the intended interface after you implement the task; the unchanged starter's car filter cannot validate a volleyball checkpoint:

```bash
python validate_video.py \
  --source data/validation/volleyball/volleyball-720p.mp4 \
  --model outputs/volleyball-train/weights/last.pt \
  --output outputs/volleyball-validation
```

Use the actual checkpoint path produced by your training run. Inspect the complete annotated video. Include screenshots at approximately 0, 5, and 10 seconds (frame indices 0, 120, and 240), and additional frames with clear successful detections. Identify missed balls, false detections, motion blur, and occlusion. Do not require a box when the ball is outside the view or fully hidden.

The supplied video has no verified bounding-box annotations. Prediction counts and confidence scores are not accuracy. Report precision, recall, or mAP only if you independently annotate evaluation frames and define the matching protocol; those annotations must never enter training.

## Submit and demonstrate

Submit your source changes, one-class YAML, label-checking script, dataset/version/split manifest, checkpoint, training log, annotated MP4, GUI screenshots, and a short result review. Include the optional generic-ball baseline if you ran it.

Acceptance criteria:

- The GUI visibly identifies the actual volleyball across multiple clear frames, with consistent labels and scores.
- All 387 frames are processed and the exported MP4 can be decoded; per-frame exported counts agree with displayed boxes.
- Pause/resume and early-stop saving work, and incomplete runs are clearly marked.
- The validation video and any matching recording are excluded from training.
- The report includes the requested timestamps and honestly explains missed or incorrect detections.

The starter kit verifies the video file and the guided car walkthrough. Volleyball model training and successful volleyball recognition are the student's work to complete.

## Sources

Dataset: [primaryws](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset) and [version 1](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset/dataset/1). Video: [Shev123, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Paris_Volley_-_Chaumont,_18_February_2014_-_26.webm). Export: [Roboflow documentation](https://docs.roboflow.com/datasets/create-and-upload/download-a-dataset). Model interface: [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/). BibTeX entries are in `docs/references.bib`.
