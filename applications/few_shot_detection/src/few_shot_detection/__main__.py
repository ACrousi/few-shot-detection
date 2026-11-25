import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Native SAM3 + Cat2Real few-shot detection (no ROS required)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", help="Override the configured SAM3 prompt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from dinov3_cases.runtime import PROJECT
    from few_shot_detection.pipeline import open_pipeline, load_settings
    from dinov3_cases.recognition.settings import validate_settings
    from PIL import Image, ImageOps
    settings = load_settings(args.config or PROJECT / "config/native.yaml")
    if args.prompt is not None:
        settings["prompt"] = validate_settings(args.prompt, settings["box_threshold"], settings["recognition_threshold"], 0)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    with open_pipeline(settings) as pipeline:
        result = {"runtime": pipeline.config["pipeline"], "image": str(args.image.resolve()),
                  "platform": pipeline.config["platform"], "size": list(image.size), "detections": pipeline.infer(image)}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
