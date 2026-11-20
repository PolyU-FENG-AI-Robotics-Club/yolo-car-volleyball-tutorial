"""Train on COCO128 only. No training image is used for validation."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import strip_optimizer
    from types import SimpleNamespace

    class TrainOnlyTrainer(DetectionTrainer):
        """Disable both periodic and mandatory final validation in this pinned release."""

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val":
                return None
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

        def label_loss_items(self, loss_items=None, prefix="train"):
            if prefix == "val":
                return [] if loss_items is None else {}
            return super().label_loss_items(loss_items, prefix)

        def validate(self):
            # val=False alone does not disable final-epoch validation in 8.3.230.
            # No validator or validation loader is invoked here.
            return {}, 0.0

        def final_eval(self):
            if self.last.exists():
                strip_optimizer(self.last)
            # With no validation there is no meaningful "best" checkpoint.
            self.best.unlink(missing_ok=True)
            self.metrics = {}
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Use the complete COCO label map. classes=[2] filters targets without remapping IDs.
    import yaml
    import ultralytics
    cfg = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
    data = yaml.safe_load(cfg.read_text())
    data.pop("download", None)
    data["path"] = str(ROOT / "data/coco128")
    validation_dir = ROOT / "data/validation"
    validation_dir.mkdir(exist_ok=True)
    data["val"] = str(validation_dir)  # Kept separate; never loaded for training/metrics.
    local_cfg = ROOT / "data/coco128-train-only.yaml"
    local_cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    print("TRAINING ONLY. Validation loaders and final evaluation are disabled. Use validate_video.py afterward.")
    YOLO(str(ROOT / "models/yolo11n.pt")).train(
        trainer=TrainOnlyTrainer,
        data=str(local_cfg), epochs=args.epochs, imgsz=320, batch=8,
        device="cpu", workers=0, classes=[2], project=str(ROOT / "outputs"),
        name="train-only", exist_ok=False, plots=False, amp=False,
        deterministic=True, seed=0, cache=False, val=False, patience=0)
