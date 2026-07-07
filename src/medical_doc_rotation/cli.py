import argparse
from pathlib import Path

from medical_doc_rotation.config import RotationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate a medical document image before downstream processing.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--triton-url", default="localhost:8000")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        "CLI wiring requires Triton adapters from Task 7 before processing real images. "
        f"Received input={args.input}, output={args.output}, triton_url={args.triton_url}, config={RotationConfig()}."
    )
