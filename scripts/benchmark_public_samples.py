import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from medical_doc_rotation.cli import read_alphabet, read_model_io
from medical_doc_rotation.decision import angular_distance, normalize_angle
from medical_doc_rotation.image_ops import load_image, rotate_image, save_image
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.triton_client import TritonCropRecognizer, TritonHttpClient, TritonOrientationClient

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class BenchmarkCase:
    source: str
    applied_rotation: float
    expected_correction: float
    predicted_angle: float
    passed: bool
    score: float
    elapsed_ms: float
    output_path: str


def iter_image_paths(samples_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in samples_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_rotations(value: str) -> list[float]:
    rotations = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rotations:
        raise ValueError("At least one rotation angle is required.")
    return rotations


def expected_correction(applied_rotation: float) -> float:
    return normalize_angle(-applied_rotation)


def angle_passes(predicted: float, expected: float, tolerance_degrees: float) -> bool:
    return angular_distance(predicted, expected) <= tolerance_degrees


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark rotation correction on public/anonymized samples.")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.add_argument("--model-io-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--tolerance-degrees", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkCase]:
    samples = iter_image_paths(args.samples_dir)[: args.limit]
    rotations = parse_rotations(args.rotations)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    client = TritonHttpClient(args.triton_url)
    model_io = read_model_io(args.model_io_path)
    preprocessor = RotationPreprocessor(
        TritonOrientationClient(client, config=args.config, model_io=model_io),
        TritonCropRecognizer(client, args.config, read_alphabet(args.dict_path), model_io=model_io),
        args.config,
    )

    cases: list[BenchmarkCase] = []
    for index, sample_path in enumerate(samples):
        original = load_image(sample_path)
        for applied in rotations:
            expected = expected_correction(applied)
            case_stem = f"{index:03d}_{sample_path.stem}_{int(applied) if applied.is_integer() else applied}"
            input_path = args.work_dir / f"{case_stem}_input.png"
            output_path = args.work_dir / f"{case_stem}_output.png"
            save_image(rotate_image(original, applied), input_path)
            result = preprocessor.process(input_path, output_path)
            best_score = max((score.score for score in result.trace.validation_scores), default=0.0)
            passed = angle_passes(result.decision.angle, expected, args.tolerance_degrees)
            cases.append(
                BenchmarkCase(
                    source=str(sample_path),
                    applied_rotation=applied,
                    expected_correction=expected,
                    predicted_angle=result.decision.angle,
                    passed=passed,
                    score=best_score,
                    elapsed_ms=result.elapsed_ms,
                    output_path=str(output_path),
                )
            )
            print(
                f"source={sample_path.name} applied={applied:.2f} expected={expected:.2f} "
                f"predicted={result.decision.angle:.2f} passed={passed} score={best_score:.3f}"
            )
    return cases


def main() -> int:
    from medical_doc_rotation.config import RotationConfig

    args = build_parser().parse_args()
    args.config = RotationConfig()
    cases = run_benchmark(args)
    passed = sum(1 for case in cases if case.passed)
    total = len(cases)
    accuracy = passed / total if total else 0.0
    print(f"summary passed={passed} total={total} accuracy={accuracy:.4f}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
