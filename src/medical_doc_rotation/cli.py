import argparse
import json
from pathlib import Path

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.triton_client import (
    ModelTensorNames,
    TritonCropRecognizer,
    TritonHttpClient,
    TritonOrientationClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate a medical document image before downstream processing.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.add_argument("--model-io-path", type=Path)
    return parser


def read_alphabet(path: Path) -> list[str]:
    return [""] + [line.strip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def read_model_io(path: Path) -> dict[str, ModelTensorNames]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: ModelTensorNames(
            input=values["input"],
            output=values["output"],
            input_shape=values.get("input_shape"),
            output_shape=values.get("output_shape"),
        )
        for name, values in raw.items()
    }


def main() -> int:
    args = build_parser().parse_args()
    config = RotationConfig()
    client = TritonHttpClient(args.triton_url)
    model_io_path = args.model_io_path or args.dict_path.parent.parent / "MODEL_IO.json"
    model_io = read_model_io(model_io_path)
    orientation_client = TritonOrientationClient(client, config, model_io=model_io)
    recognizer = TritonCropRecognizer(client, config, read_alphabet(args.dict_path), model_io=model_io)
    preprocessor = RotationPreprocessor(orientation_client, recognizer, config)
    result = preprocessor.process(args.input, args.output)
    print(
        f"output={result.output_path} angle={result.decision.angle:.2f} "
        f"rotate={result.decision.should_rotate} elapsed_ms={result.elapsed_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
