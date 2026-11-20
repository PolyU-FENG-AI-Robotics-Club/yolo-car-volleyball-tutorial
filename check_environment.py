"""Print the exact runtime used by the tutorial."""
import json
import platform
import sys
import os
from pathlib import Path
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import torch
import torchvision
import ultralytics
import cv2

if __name__ == "__main__":
    info = {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__, "cuda_available": torch.cuda.is_available(),
            "default_device": "cpu"}
    root = Path(__file__).resolve().parent
    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs/environment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
