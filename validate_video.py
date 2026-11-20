"""Independent road-video validation with a desktop interface or --headless mode."""
import argparse
import json
from pathlib import Path
import cv2
from detect import main as detect, ROOT


def prepare_video():
    source = ROOT / "data/validation/traffic.mp4"
    target = source.with_name("traffic-720p.mp4")
    if target.is_file():
        return target
    if not source.is_file():
        raise RuntimeError("Run python prepare_data.py first, or extract the supplied validation-video ZIP into the project folder.")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Cannot open validation video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("MP4 writer unavailable.")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, (1280, 720)))
            n += 1
    finally:
        cap.release()
        writer.release()
    if n != 538:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete video: decoded {n} of 538 frames.")
    return target


def run_validation(model, output, source=None, conf=0.25, frame_callback=None, status_callback=None):
    """Save results; callbacks keep the desktop responsive without changing inference."""
    source = Path(source).expanduser().resolve() if source else prepare_video()
    if not source.is_file() or source.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Select an existing video file, not a training image.")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Cannot decode the selected video.")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if status_callback:
        status_callback(f"Loading model and processing {expected} video frames...")
    output = detect(["--source", str(source), "--model", str(model), "--output", str(output),
                     "--conf", str(conf)], frame_callback=frame_callback)
    if status_callback:
        status_callback("Saving the annotated MP4 and checking the exported frames...")
    avi = next(output.glob("*.avi"))
    cap = cv2.VideoCapture(str(avi))
    fps = cap.get(cv2.CAP_PROP_FPS) or source_fps or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp4 = output / "validated-traffic.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not read AVI or create MP4 output.")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n in (0, 125, 250, 375, 500):
                cv2.imwrite(str(output / f"frame-{n:04}.jpg"), frame)
            writer.write(frame)
            n += 1
    finally:
        cap.release()
        writer.release()
    summary = json.loads((output / "summary.json").read_text())
    if n != summary["frames_or_images"]:
        raise RuntimeError("Exported video frame count does not match the detection records.")
    completed = expected > 0 and n == expected
    supplied_video = source in {ROOT / "data/validation/traffic.mp4", ROOT / "data/validation/traffic-720p.mp4"}
    summary.update({"validation_type": "Visual video validation; no ground-truth accuracy metrics",
                    "source_documentation": "https://supervision.roboflow.com/assets/" if supplied_video else None,
                    "training_data": "COCO128 images only",
                    "validation_data": "Roboflow vehicles.mp4; no frames used for local training" if supplied_video else str(source),
                    "expected_frames": expected, "completed": completed,
                    "duration_seconds": n / fps, "decoded_output_frames": n,
                    "annotated_mp4": str(mp4)})
    (output / "validation-report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Video validation {'complete' if completed else 'stopped; partial results saved'}: {mp4}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/yolo11n.pt"))
    parser.add_argument("--output", default=str(ROOT / "outputs/validation-video"))
    parser.add_argument("--source", help="Independent local video; defaults to the provided highway clip.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true", help="Run without opening the desktop interface.")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1].")
    if args.headless:
        run_validation(args.model, args.output, args.source, args.conf)
    else:
        from validation_ui import launch
        launch(args)
