from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roadsigns.common import read_yaml
from roadsigns.image_tools.geometry import windows
from roadsigns.image_tools.labels import read_labels
from roadsigns.ontology import Ontology


def draw_boxes(image, classes, boxes, names, color="lime"):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=max(12, image.width // 170))
    for cls, box in zip(classes, boxes):
        draw.rectangle(tuple(box), outline=color, width=max(2, image.width // 900))
        draw.text(
            (float(box[0]), max(0, float(box[1]) - font.size - 2)),
            names[int(cls)],
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Preview source or prepared YOLO image labels")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    names = parser.add_mutually_exclusive_group(required=True)
    names.add_argument("--ontology", type=Path, help="Original 401-class ontology CSV")
    names.add_argument("--data", type=Path, help="Prepared data.yaml")
    parser.add_argument("--tiles", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    class_names = (
        [r["source_name"] for r in Ontology(args.ontology).rows]
        if args.ontology
        else read_yaml(args.data)["names"]
    )
    image = Image.open(args.image).convert("RGB")
    classes, boxes = read_labels(args.labels, *image.size, len(class_names))
    result = draw_boxes(image, classes, boxes, class_names)
    if args.tiles:
        draw = ImageDraw.Draw(result)
        for index, roi in enumerate(windows(*image.size)):
            draw.rectangle(roi, outline="cyan", width=2)
            draw.text((roi[0] + 4, roi[1] + 4), str(index), fill="cyan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
