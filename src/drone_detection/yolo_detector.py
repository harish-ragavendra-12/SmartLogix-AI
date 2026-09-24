"""
SmartLogix AI
YOLO Object Detection Module

This module provides a reusable YOLO inference wrapper.

Note:
The current project does not contain a labeled drone-damage image
dataset. Therefore, this implementation demonstrates the YOLO
computer-vision pipeline using a pretrained YOLO model.

A custom drone-damage model can be plugged in later without changing
the inference interface.
"""

from pathlib import Path

from ultralytics import YOLO


# ------------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "yolo"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "yolo"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class YOLODetector:
    """Reusable YOLO inference wrapper."""

    def __init__(self, model_name="yolo11n.pt"):

        self.model_path = MODEL_DIR / model_name

        print("=" * 70)
        print("LOADING YOLO MODEL")
        print("=" * 70)

        print(f"Model path:")
        print(self.model_path)

        print("\nLoading model...")

        self.model = YOLO(str(self.model_path))

        print("YOLO model loaded successfully.")

    def predict(self, image_path, confidence=0.25):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found:\n{image_path}"
            )

        print("\n" + "=" * 70)
        print("YOLO INFERENCE")
        print("=" * 70)

        print(f"Input image:")
        print(image_path)

        print(f"\nConfidence threshold: {confidence}")

        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            save=True,
            project=str(OUTPUT_DIR),
            name="inference",
            exist_ok=True,
            verbose=True
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence_score = float(
                    box.conf[0]
                )

                coordinates = [
                    round(float(value), 2)
                    for value in box.xyxy[0]
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.model.names[
                            class_id
                        ],
                        "confidence": round(
                            confidence_score,
                            4
                        ),
                        "bounding_box": coordinates
                    }
                )

        output_path = (
            OUTPUT_DIR
            / "inference"
            / image_path.name
        )

        return {
            "status": "success",
            "model": self.model_path.name,
            "image": str(image_path),
            "detection_count": len(detections),
            "detections": detections,
            "output_path": str(output_path)
        }