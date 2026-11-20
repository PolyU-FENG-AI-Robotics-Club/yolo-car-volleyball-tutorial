"""Download verified official assets and prepare a compact, reproducible demo."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
BUNDLED_VIDEO_SHA256 = "d5ea1db49cf47e238746ea42ebf89a5b827623e4e27458996df969708ff011c7"
ASSETS = {
    "models/yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "data/coco128.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"),
    "data/validation/traffic.mp4": (
        "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
        "ac81100d9310bd4e9c02bc0b13b6492781d009742ced347766b2601be3c44ad4"),
}


def main():
    bundled_video = ROOT / "data/validation/traffic-720p.mp4"
    use_bundled_video = bundled_video.is_file()
    if use_bundled_video:
        if hashlib.sha256(bundled_video.read_bytes()).hexdigest() != BUNDLED_VIDEO_SHA256:
            raise RuntimeError("Bundled video checksum mismatch. Re-extract the supplied video ZIP, or remove traffic-720p.mp4 and retry.")
        print("Verified bundled 720p validation video; no 4K download needed.")
    for name, (url, expected) in ASSETS.items():
        if use_bundled_video and name == "data/validation/traffic.mp4":
            continue
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {name} ...", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Checksum mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {path}. Remove this file and retry.")
    target = ROOT / "data"
    with zipfile.ZipFile(target / "coco128.zip") as archive:
        for member in archive.infolist():
            if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                raise RuntimeError("Unsafe archive path")
        archive.extractall(target)
    images = target / "coco128/images/train2017"
    labels = target / "coco128/labels/train2017"
    positives = []
    for label in sorted(labels.glob("*.txt")):
        if not (images / (label.stem + ".jpg")).is_file():
            continue
        cars = [line for line in label.read_text().splitlines() if line.split()[0] == "2"]
        if cars:
            positives.append({"image": label.stem + ".jpg", "cars": len(cars)})
    demo = target / "cars"
    demo.mkdir(exist_ok=True)
    for item in positives:
        shutil.copy2(images / item["image"], demo / item["image"])
    # The COCO street scene has multiple visible cars; verify the ID exists.
    preferred = images / "000000000094.jpg"
    sample = preferred if any(x["image"] == preferred.name for x in positives) else images / positives[0]["image"]
    shutil.copy2(sample, target / "sample.jpg")
    manifest = {"dataset": "COCO128", "total_images": len(list(images.glob("*.jpg"))),
                "car_images": len(positives), "car_annotations": sum(x["cars"] for x in positives),
                "sample_original": sample.name, "car_class_id": 2, "images": positives,
                "purpose": "Pipeline demonstration only; COCO128 comes from COCO train2017.",
                "assets": {k: {"url": u, "sha256": h} for k, (u, h) in ASSETS.items()}}
    manifest["bundled_validation_video"] = {
        "used": use_bundled_video,
        "path": "data/validation/traffic-720p.mp4",
        "sha256": BUNDLED_VIDEO_SHA256,
        "source": ASSETS["data/validation/traffic.mp4"][0],
        "transformation": "Resized to 1280x720; all 538 frames retained at 25 FPS",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("images", "assets")}, indent=2))


if __name__ == "__main__":
    main()
