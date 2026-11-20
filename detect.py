"""Car-only YOLO detection for an image, folder, video, or webcam."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "data/sample.jpg"))
    parser.add_argument("--model", default=str(ROOT / "models/yolo11n.pt"))
    parser.add_argument("--output", default=str(ROOT / "outputs/demo"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true", help="Open a preview window; requires a desktop.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N results; 0 means unlimited.")
    args = parser.parse_args(argv)
    if not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1].")
    if args.imgsz < 32 or args.max_frames < 0:
        parser.error("--imgsz must be >= 32 and --max-frames must be >= 0.")
    if not Path(args.model).is_file():
        parser.error("Model not found. Run python prepare_data.py first.")
    if not args.source.isdecimal() and not Path(args.source).exists():
        parser.error("Source not found. Supply a local file, folder, or webcam index such as 0.")
    return args


def main(argv=None, frame_callback=None):
    args = parse_args(argv)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu.")
    model = YOLO(args.model)
    car_ids = [i for i, name in model.names.items() if name == "car"]
    if not car_ids:
        raise SystemExit("This model has no class named 'car'. Choose a detection model with that class.")
    output = Path(args.output).resolve()
    original = output
    suffix = 2
    while output.exists():
        output = original.with_name(f"{original.name}{suffix}")
        suffix += 1
    output.mkdir(parents=True)
    count = frames = 0
    inference_ms = 0.0
    start = time.perf_counter()
    results = model.predict(source=int(args.source) if args.source.isdecimal() else args.source,
                            classes=car_ids, conf=args.conf, imgsz=args.imgsz, device=args.device,
                            stream=True, save=True, save_txt=True, save_conf=True,
                            project=str(output.parent), name=output.name, exist_ok=True,
                            show=args.show, verbose=False)
    try:
        with (output / "detections.csv").open("w", newline="") as f, (output / "frames.jsonl").open("w") as j:
            writer = csv.DictWriter(f, fieldnames=["frame", "source", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for result in results:
                frames += 1
                inference_ms += result.speed.get("inference", 0)
                boxes = []
                for box in result.boxes:
                    xyxy = [round(float(x), 2) for x in box.xyxy[0].tolist()]
                    row = dict(frame=frames, source=str(result.path), class_id=int(box.cls.item()),
                               class_name="car", confidence=round(float(box.conf.item()), 6),
                               **dict(zip(["x1", "y1", "x2", "y2"], xyxy)))
                    writer.writerow(row)
                    boxes.append(row)
                count += len(boxes)
                j.write(json.dumps({"frame": frames, "source": str(result.path), "car_count": len(boxes), "boxes": boxes}) + "\n")
                print(f"Frame {frames}: {Path(result.path).name} | cars={len(boxes)}", flush=True)
                if frame_callback is not None and frame_callback(result, frames) is False:
                    break
                if args.max_frames and frames >= args.max_frames:
                    break
    finally:
        results.close()
        # Ultralytics' generator may be closed early for webcam/max-frames use.
        for writer in getattr(model.predictor, "vid_writer", {}).values():
            writer.release()
        dataset = getattr(model.predictor, "dataset", None)
        cap = getattr(dataset, "cap", None)
        if cap is not None:
            cap.release()
    summary = {"model": str(Path(args.model).resolve()), "source": args.source,
               "device": args.device, "confidence_threshold": args.conf, "image_size": args.imgsz,
               "car_class_ids": car_ids, "frames_or_images": frames, "car_detections": count,
               "elapsed_seconds": round(time.perf_counter() - start, 3),
               "mean_inference_ms": round(inference_ms / frames, 3) if frames else None,
               "output_directory": str(output),
               "note": "Video counts are per-frame detections, not unique vehicles. Confidence is not accuracy."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    main()
