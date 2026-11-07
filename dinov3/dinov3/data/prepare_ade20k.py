"""Create the split lists expected by the ADE20K loader from an extracted dataset."""

import argparse
from pathlib import Path


def prepare(root):
    root = Path(root)
    for split, folder, expected in (("train", "training", 20210), ("val", "validation", 2000)):
        images = sorted((root / "images" / folder).glob("*.jpg"))
        if len(images) != expected:
            raise ValueError(f"Expected {expected} {split} images, found {len(images)}")
        rows = [image.relative_to(root / "images").as_posix() for image in images]
        for image in rows:
            mask = root / "annotations" / Path(image).with_suffix(".png")
            if not mask.is_file():
                raise FileNotFoundError(mask)
        target = root / f"ADE20K_object150_{split}.txt"
        if target.exists() and sorted(target.read_text().splitlines()) != rows:
            raise ValueError(f"Existing split list differs from the extracted dataset: {target}")
        target.write_text("\n".join(rows) + "\n")
        print(f"{target}: {len(rows)} image/mask pairs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.root)


if __name__ == "__main__":
    main()
