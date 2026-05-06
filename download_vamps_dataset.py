import argparse
import io
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset  # type: ignore[import-untyped]
from PIL import Image

DEFAULT_SPLITS = ["Konkour_EN", "Konkour_FA", "Synth_EN", "Synth_FA"]


def parse_splits(split_arg: str) -> List[str]:
    value = (split_arg or "all").strip()
    if value.lower() == "all":
        return list(DEFAULT_SPLITS)
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("--split must be 'all' or a comma-separated list of split names.")
    return splits


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "image"


def _write_image_from_bytes(raw: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as img:
        img.save(target)


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _detect_ext(path_hint: Optional[str], default_ext: str = "png") -> str:
    if isinstance(path_hint, str) and path_hint.strip():
        suffix = Path(path_hint).suffix.lower().lstrip(".")
        if suffix in {"png", "jpg", "jpeg", "webp", "gif"}:
            return suffix
    return default_ext


def _content_type_from_ext(ext: str) -> str:
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    return f"image/{ext}"


def _save_image_field(
    image_value: Any,
    images_dir: Path,
    img_counter: int,
) -> tuple[Optional[Dict[str, str]], int]:
    if image_value is None:
        return None, img_counter

    ext = "png"
    out_bytes: Optional[bytes] = None
    path_hint: Optional[str] = None

    if isinstance(image_value, Image.Image):
        bio = io.BytesIO()
        image_value.save(bio, format="PNG")
        out_bytes = bio.getvalue()
        ext = "png"
    elif isinstance(image_value, dict):
        if isinstance(image_value.get("path"), str):
            path_hint = image_value.get("path")
        ext = _detect_ext(path_hint, default_ext="png")
        raw = image_value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            out_bytes = bytes(raw)
    else:
        return None, img_counter

    if out_bytes is None:
        return None, img_counter

    digest = hashlib.sha1(out_bytes).hexdigest()[:12]
    file_name = f"img_{img_counter:04d}_{digest}.{ext}"
    out_path = images_dir / file_name
    _write_image_from_bytes(out_bytes, out_path)

    image_obj = {
        "content_type": _content_type_from_ext(ext),
        "ext": ext,
        "path": str(Path("images") / file_name),
    }
    return image_obj, img_counter + 1


def _build_record(example: Dict[str, Any], split: str, images_dir: Path, img_counter: int) -> tuple[Dict[str, Any], int]:
    # ID fallback strategy: keep canonical string if present, else deterministic split+sample_id.
    raw_id = example.get("id")
    if raw_id is None:
        raw_id = example.get("source_id")
    if raw_id is None:
        raw_id = f"{split}-{example.get('sample_id', 'unknown')}"
    sample_id = str(raw_id)

    original_image, img_counter = _save_image_field(example.get("question_image"), images_dir, img_counter)

    options: List[Dict[str, Any]] = []
    for idx in range(1, 5):
        opt_img, img_counter = _save_image_field(example.get(f"option_{idx}_image"), images_dir, img_counter)
        options.append(
            {
                "label": str(idx),
                "text": _norm_text(example.get(f"option_{idx}_text")),
                "image": opt_img,
            }
        )

    record: Dict[str, Any] = {
        "id": sample_id,
        "qnum": example.get("qnum"),
        "question_text": _norm_text(example.get("question_text")),
        "original_image": original_image,
        "question_images": [],
        "visualization_images": [],
        "options": options,
        "answer": _norm_text(example.get("answer")),
        "difficulty": None,
        "provenance": {
            "docx": None,
            "start_paragraph_index": None,
        },
    }
    return record, img_counter


def _convert_split(dataset_name: str, split: str, out_dir: Path, max_examples: Optional[int] = None) -> None:
    ds = load_dataset(dataset_name, split=split)
    split_dir = out_dir / split
    images_dir = split_dir / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    image_counter = 0
    for idx, example in enumerate(ds):
        if max_examples is not None and idx >= max_examples:
            break
        if not isinstance(example, dict):
            raise ValueError(f"Unexpected non-dict record in split {split} at index {idx}.")
        converted, image_counter = _build_record(example, split, images_dir, image_counter)
        records.append(converted)

    data_json_path = split_dir / "data.json"
    data_json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {split}: {len(records)} records -> {data_json_path}")


def download_vamps(dataset_name: str, splits: List[str], out_dir: Path, max_examples: Optional[int] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        _convert_split(dataset_name=dataset_name, split=split, out_dir=out_dir, max_examples=max_examples)
    print(f"Download + extraction complete at {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download VAMPS dataset splits from Hugging Face with the original folder layout "
            "(split/data.json + split/images/...)."
        )
    )
    ap.add_argument(
        "--dataset-name",
        default="VAMPSBenchmark/VAMPS",
        help="Hugging Face dataset name (default: VAMPSBenchmark/VAMPS).",
    )
    ap.add_argument(
        "--split",
        default="all",
        help=(
            "Which split(s) to download. Use 'all' (default) or a comma-separated list from: "
            "Konkour_EN,Konkour_FA,Synth_EN,Synth_FA."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="data",
        help="Output directory root (default: data).",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per split for quick testing.",
    )
    args = ap.parse_args()
    splits = parse_splits(args.split)
    download_vamps(
        dataset_name=args.dataset_name,
        splits=splits,
        out_dir=Path(args.out_dir),
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

