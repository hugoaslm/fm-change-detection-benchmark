"""Dataset loading, validation, and synthetic generation for LEVIR-CD."""

import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ChangeSample(TypedDict):
    sample_id: str
    scene_id: str
    image_t1: Tensor  # [3, H, W], float32 in [0, 1]
    image_t2: Tensor  # [3, H, W], float32 in [0, 1]
    change_mask: Tensor  # [H, W], bool


def extract_scene_id(sample_id: str) -> str:
    """Extract scene_id from sample_id for scene-level cluster bootstrapping."""
    base = Path(sample_id).stem
    # Handle LEVIR-CD naming formats (e.g., scene_001_tile_01 or levir_001_x_y or train_1_0_0)
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] in ("scene", "levir", "train", "val", "test", "build"):
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else base


class LEVIRCDDataset(Dataset):
    """Dataset loader for LEVIR-CD change detection dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        input_size: int = 252,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.input_size = input_size

        self.dir_a = self.root / "A"
        self.dir_b = self.root / "B"
        self.dir_label = self.root / "label"
        self.list_file = self.root / "list" / f"{split}.txt"

        if not self.list_file.exists():
            raise FileNotFoundError(f"List file not found at {self.list_file}")

        with open(self.list_file, "r", encoding="utf-8") as f:
            self.file_list = [line.strip() for line in f if line.strip()]
        if not self.file_list:
            raise ValueError(f"Split list is empty: {self.list_file}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> ChangeSample:
        filename = self.file_list[idx]
        stem = Path(filename).stem

        path_a = self.dir_a / filename
        path_b = self.dir_b / filename
        path_label = self.dir_label / filename

        if not path_a.exists():
            # Try appending .png if extension missing in list
            path_a = self.dir_a / f"{stem}.png"
            path_b = self.dir_b / f"{stem}.png"
            path_label = self.dir_label / f"{stem}.png"

        with Image.open(path_a) as image:
            img_a = image.convert("RGB")
        with Image.open(path_b) as image:
            img_b = image.convert("RGB")
        with Image.open(path_label) as image:
            label = image.convert("L")

        if img_a.size != img_b.size or img_a.size != label.size:
            raise ValueError(
                f"Unaligned sample '{filename}': A={img_a.size}, B={img_b.size}, label={label.size}"
            )

        # Convert images to tensors [3, H, W] in [0, 1]
        arr_a = np.array(img_a, dtype=np.float32) / 255.0
        arr_b = np.array(img_b, dtype=np.float32) / 255.0
        t1 = torch.from_numpy(arr_a).permute(2, 0, 1)
        t2 = torch.from_numpy(arr_b).permute(2, 0, 1)

        # Convert label to bool mask [H, W]
        arr_label = np.array(label, dtype=np.uint8)
        mask = torch.from_numpy(arr_label > 127)  # True for change

        scene_id = extract_scene_id(stem)

        return ChangeSample(
            sample_id=stem,
            scene_id=scene_id,
            image_t1=t1,
            image_t2=t2,
            change_mask=mask,
        )


def validate_dataset_layout(
    root: str | Path, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, int]:
    """Validate LEVIR-CD directory layout and return sample counts per split."""
    path = Path(root)
    required_dirs = [path / "A", path / "B", path / "label", path / "list"]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Required directory missing: {d}")

    counts = {}
    split_names: dict[str, set[str]] = {}
    split_scenes: dict[str, set[str]] = {}
    for split in splits:
        list_file = path / "list" / f"{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(f"Required split list file missing: {list_file}")
        with open(list_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            raise ValueError(f"Split list is empty: {list_file}")
        if len(lines) != len(set(lines)):
            raise ValueError(f"Duplicate filenames found in {list_file}")
        split_names[split] = set(lines)
        split_scenes[split] = {extract_scene_id(name) for name in lines}
        for filename in lines:
            stem = Path(filename).stem
            candidates = [filename, f"{stem}.png"]
            for directory in ("A", "B", "label"):
                if not any((path / directory / candidate).is_file() for candidate in candidates):
                    raise FileNotFoundError(
                        f"Referenced sample '{filename}' missing from {path / directory}"
                    )
        counts[split] = len(lines)

    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = split_names[left] & split_names[right]
            if overlap:
                example = min(overlap)
                raise ValueError(f"Split leakage between {left} and {right}; example: {example}")
            scene_overlap = split_scenes[left] & split_scenes[right]
            if scene_overlap:
                example = min(scene_overlap)
                raise ValueError(
                    f"Scene-level split leakage between {left} and {right}; example: {example}"
                )

    return counts


def compute_dataset_manifest_hash(
    root: str | Path, splits: tuple[str, ...] = ("train", "val", "test")
) -> str:
    """Hash split manifests and referenced relative names without reading image bytes."""
    path = Path(root)
    digest = hashlib.sha256()
    for split in splits:
        list_path = path / "list" / f"{split}.txt"
        digest.update(split.encode())
        digest.update(list_path.read_bytes())
    return digest.hexdigest()[:16]


def generate_synthetic_dataset(
    output_root: str | Path,
    num_scenes: int = 4,
    tiles_per_scene: int = 2,
    image_size: int = 256,
    seed: int = 42,
) -> Path:
    """Generate synthetic LEVIR-CD dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    root = Path(output_root)

    dir_a = root / "A"
    dir_b = root / "B"
    dir_label = root / "label"
    dir_list = root / "list"

    for d in [dir_a, dir_b, dir_label, dir_list]:
        d.mkdir(parents=True, exist_ok=True)

    split_files = {"train": [], "val": [], "test": []}
    splits = ["train", "val", "test"]

    sample_counter = 0
    for s_idx in range(num_scenes):
        scene_id = f"scene_{s_idx:03d}"
        split = splits[s_idx % len(splits)]

        for t_idx in range(tiles_per_scene):
            filename = f"{scene_id}_tile_{t_idx:02d}.png"
            sample_counter += 1

            # Generate synthetic background and change patch
            img_a_arr = rng.integers(50, 200, size=(image_size, image_size, 3), dtype=np.uint8)
            img_b_arr = img_a_arr.copy()

            label_arr = np.zeros((image_size, image_size), dtype=np.uint8)

            # 50% of tiles contain change
            if sample_counter % 2 == 1:
                cx, cy = image_size // 4, image_size // 4
                rw, rh = image_size // 2, image_size // 2
                img_b_arr[cy : cy + rh, cx : cx + rw] = rng.integers(
                    200, 255, size=(rh, rw, 3), dtype=np.uint8
                )
                label_arr[cy : cy + rh, cx : cx + rw] = 255

            Image.fromarray(img_a_arr).save(dir_a / filename)
            Image.fromarray(img_b_arr).save(dir_b / filename)
            Image.fromarray(label_arr).save(dir_label / filename)

            split_files[split].append(filename)

    for split, files in split_files.items():
        with open(dir_list / f"{split}.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{fname}\n" for fname in files)

    return root
