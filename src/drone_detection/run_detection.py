"""
SmartLogix AI
Computer Vision - YOLO Demonstration

Runs a lightweight YOLO inference using the sample image
already included with the installed Ultralytics package.
"""

from pathlib import Path
import sys


# ------------------------------------------------------------------
# MAKE PROJECT SRC IMPORTABLE
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------------
# IMPORT YOLO MODULE
# ------------------------------------------------------------------

from drone_detection.yolo_detector import YOLODetector


def main():

    print("\n")
    print("=" * 70)
    print("SMARTLOGIX AI - COMPUTER VISION")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------------
    # SAMPLE IMAGE
    # --------------------------------------------------------------

    image_path = (
        PROJECT_ROOT
        / ".venv"
        / "Lib"
        / "site-packages"
        / "ultralytics"
        / "assets"
        / "bus.jpg"
    )

    print("\nSample image:")
    print(image_path)

    if not image_path.exists():

        print("\nERROR:")
        print("Ultralytics sample image was not found.")

        print("\nExpected:")
        print(image_path)

        return

    print("\nSample image exists: YES")

    # --------------------------------------------------------------
    # LOAD DETECTOR
    # --------------------------------------------------------------

    try:

        detector = YOLODetector(
            model_name="yolo11n.pt"
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("YOLO MODEL LOADING FAILED")
        print("=" * 70)

        print(type(error).__name__)
        print(str(error))

        return

    # --------------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------------

    try:

        result = detector.predict(
            image_path=image_path,
            confidence=0.25
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("YOLO INFERENCE FAILED")
        print("=" * 70)

        print(type(error).__name__)
        print(str(error))

        return

    # --------------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO DETECTION RESULT")
    print("=" * 70)

    print(f"\nStatus          : {result['status']}")
    print(f"Model           : {result['model']}")
    print(
        f"Detection Count : "
        f"{result['detection_count']}"
    )

    print("\nDetections")
    print("-" * 70)

    if result["detections"]:

        for index, detection in enumerate(
            result["detections"],
            start=1
        ):

            print(f"\nDetection {index}")

            print(
                f"Class       : "
                f"{detection['class_name']}"
            )

            print(
                f"Confidence  : "
                f"{detection['confidence']:.2%}"
            )

            print(
                f"Bounding Box: "
                f"{detection['bounding_box']}"
            )

    else:

        print("No objects detected.")

    print("\nAnnotated output:")
    print(result["output_path"])

    # --------------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPUTER VISION STATUS")
    print("=" * 70)

    if result["status"] == "success":

        print("YOLO INFERENCE : SUCCESS")

    else:

        print("YOLO INFERENCE : FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()