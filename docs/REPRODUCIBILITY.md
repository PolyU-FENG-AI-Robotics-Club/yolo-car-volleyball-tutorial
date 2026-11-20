# Verified student walkthrough

Checked on 20 November 2026 using a newly extracted student kit and a newly created Conda environment on Linux x86-64 with a graphical desktop. No existing result directories or preinstalled project environment were copied into the test folder.

| Tutorial step | Observed result |
|---|---|
| Extract student ZIP | Code, model, COCO128 archive, and 720p video present |
| Create environment | Python 3.10.22 in the new project prefix |
| Install CPU dependencies | PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, Ultralytics 8.3.230 |
| Check dependencies | `pip check`: no broken requirements |
| Prepare assets | Checksums pass; 128 images, 12 car-containing images |
| Run image detection | 2 car predictions on the supplied sample |
| Run the image folder | 12 images processed |
| Start GUI validation | All 538 frames processed with the pretrained model |
| Inspect saved MP4 | All 538 exported frames decoded successfully |
| Check exported labels | 1,278 per-frame predictions, all labeled `car` |

The final count is per-frame detections, not unique cars or an accuracy score. The video is independent of local training. No training is required to complete the guided car-detection task.

Open `outputs/validation-video/validated-traffic.mp4`. In `validation-report.json`, check `completed: true` and `decoded_output_frames: 538`. Repeated runs use a numbered output-folder suffix.

The prerequisites are a Linux graphical desktop, Conda, network access for dependency installation, and sufficient disk space. Windows, macOS, and physical webcam capture were not part of this reproduction check.

## Volleyball assignment materials (v1.1.0)

The guided car application code is unchanged from the verified release above. The additional volleyball MP4 was decoded in full: 387 frames at 24000/1001 FPS, 1280×720. The package includes provenance and SHA-256 records. Volleyball training and recognition remain the independent student task and are not claimed as implemented or verified.
