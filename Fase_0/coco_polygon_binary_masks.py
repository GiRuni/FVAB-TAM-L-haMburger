from pathlib import Path

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise ImportError(
        "pycocotools is required. Install it with: pip install pycocotools"
    ) from exc


TARGET_IMG_IDS_STR = [
    "000000000785",
    "000000002149",
    "000000007784",
    "000000010764",
    "000000011051",
]


DEFAULT_INSTANCES_JSON = Path(r"/content/FVAB-TAM-L-haMburger/tam-logit-lenses/ll_tam/data/coco2014/annotations/instances_minival2014.json")
DEFAULT_OUTPUT_DIR = Path(r"/content/FVAB-TAM-L-haMburger/tam-logit-lenses/ll_tam/masks")


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find instances_minival2014.json. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_output_dir(instances_json: Path) -> Path:
    preferred = [
        DEFAULT_OUTPUT_DIR,
        instances_json.parent.parent / "masks",
        Path.cwd() / "masks",
    ]
    for candidate in preferred:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("Could not create any writable output directory.")


def _polygon_segmentation_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    rles = mask_utils.frPyObjects(segmentation, height, width)
    if isinstance(rles, dict):
        rles = [rles]
    merged = mask_utils.merge(rles)  # type: ignore[arg-type]
    decoded = mask_utils.decode(merged)
    return decoded.astype(bool)


def sanitize_filename(name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name.strip().lower())
    return safe_name or "mask"


def build_binary_mask_for_annotation(coco: COCO, ann: dict, height: int, width: int) -> np.ndarray | None:
    segmentation = ann.get("segmentation")

    if not isinstance(segmentation, list):
        return None

    if not segmentation:
        return None

    return _polygon_segmentation_to_mask(segmentation, height, width)


def build_category_name(coco: COCO, category_id: int) -> str:
    category = coco.loadCats([category_id])[0]
    return sanitize_filename(category["name"])


def save_png(mask_uint8: np.ndarray, out_path: Path) -> None:
    from PIL import Image

    img = Image.fromarray(mask_uint8, mode="L")
    img.save(out_path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    instances_json = resolve_existing_path(
        DEFAULT_INSTANCES_JSON,
        script_dir / "instances_minival2014.json",
        script_dir / "annotations" / "instances_minival2014.json",
        Path.cwd() / "instances_minival2014.json",
        Path.cwd() / "annotations" / "instances_minival2014.json",
        Path("/content/drive/MyDrive/data/coco2014/annotations/instances_minival2014.json"),
    )
    output_dir = resolve_output_dir(instances_json)

    target_img_ids = [int(x) for x in TARGET_IMG_IDS_STR]

    coco = COCO(str(instances_json))

    try:
        import PIL  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Pillow is required to save PNG masks. Install it with: pip install pillow") from exc

    for image_id in target_img_ids:
        img_id_str = f"{image_id:012d}"
        img_info = coco.loadImgs([image_id])[0]
        height = img_info["height"]
        width = img_info["width"]
        image_output_dir = output_dir / img_id_str
        image_output_dir.mkdir(parents=True, exist_ok=True)

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        category_counts: dict[str, int] = {}

        for ann in anns:
            category_id = ann.get("category_id")
            if not isinstance(category_id, int):
                continue

            mask_bool = build_binary_mask_for_annotation(coco, ann, height, width)
            if mask_bool is None:
                continue

            category_name = build_category_name(coco, category_id)
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
            suffix = "" if category_counts[category_name] == 1 else f"_{category_counts[category_name]}"
            mask_png = mask_bool.astype(np.uint8) * 255
            save_png(mask_png, image_output_dir / f"{category_name}{suffix}.png")

    print(f"Done. Wrote masks to: {output_dir}")


if __name__ == "__main__":
    main()
