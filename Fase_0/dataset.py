import csv
import argparse
import json
from pathlib import Path


DEFAULT_CAPTIONS_JSON = Path(r"/content/FVAB-TAM-L-haMburger/tam-logit-lenses/ll_tam/instances_minival2014.json")
DEFAULT_INSTANCES_JSON = Path(r"/content/FVAB-TAM-L-haMburger/tam-logit-lenses/ll_tam/data/coco2014/annotations/instances_minival2014.json")
DEFAULT_OUTPUT_CSV = Path(r"captions_selected.csv")


TARGET_IMG_IDS_STR = [
    "000000000785",
    "000000002149",
    "000000007784",
    "000000010764",
    "000000011051",
    "000000013004",
    "000000013597",
    "000000020059",
    "000000021604",
    "000000023359",
    "000000032081",
    "000000079188",
    "000000568690",
    "000000578967",
    "000000579091",
    "000000579818",
    "000000579893",
    "000000580757",
    "000000581206",
    "000000581317",
    "000000581615",
]


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the captions JSON file. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select COCO instance categories for target image IDs.")
    parser.add_argument(
        "--instances-json",
        type=Path,
        default=None,
        help="Path to instances_minival2014.json",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Where to write the selected CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_img_ids_int = {int(x) for x in TARGET_IMG_IDS_STR}

    script_dir = Path(__file__).resolve().parent
    instances_json = args.instances_json or resolve_existing_path(
        DEFAULT_INSTANCES_JSON,
        script_dir / "instances_minival2014.json",
        script_dir / "annotations" / "instances_minival2014.json",
        Path.cwd() / "instances_minival2014.json",
        Path.cwd() / "annotations" / "instances_minival2014.json",
        Path("/content/FVAB-TAM-L'haMburger/Fase_0/tam-logit-lenses/ll_tam/data/coco2014/annotations/instances_minival2014.json"),
    )
    output_csv = args.output_csv or DEFAULT_OUTPUT_CSV

    with instances_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_name_by_id = {cat.get("id"): cat.get("name", "").strip() for cat in categories}

    ann_entries_by_image_id = {}

    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id in target_img_ids_int and category_id in category_name_by_id:
            obj_name = category_name_by_id[category_id]
            ann_id = ann.get("id")
            if obj_name and ann_id is not None:
                ann_entries_by_image_id.setdefault(image_id, []).append((ann_id, obj_name))

    rows = []

    for image_id in sorted(target_img_ids_int):
        img_id = f"{image_id:012d}"
        ann_entries = sorted(ann_entries_by_image_id.get(image_id, []), key=lambda x: x[0])

        row = {
            "img_id": img_id,
            "path": f"coco2014/image/{img_id}.jpg",
            "obj_main": "",
            "ann_id_main": "",
            "obj_n": "",
            "ann_id_n": "",
        }

        if ann_entries:
            main_ann_id, main_obj = ann_entries[0]
            row["obj_main"] = main_obj
            row["ann_id_main"] = main_ann_id

            remaining = ann_entries[1:]
            row["obj_n"] = ", ".join(obj_name for _, obj_name in remaining)
            row["ann_id_n"] = ", ".join(str(ann_id) for ann_id, _ in remaining)

        rows.append(row)

    rows.sort(key=lambda r: r["img_id"])

    fieldnames = ["img_id", "path", "obj_main", "ann_id_main", "obj_n", "ann_id_n"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
