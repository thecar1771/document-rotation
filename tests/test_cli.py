from pathlib import Path

from medical_doc_rotation import cli
from medical_doc_rotation.types import (
    AngleCandidate,
    AngleScore,
    OrientationScores,
    RotationDecision,
    RotationResult,
    RotationTrace,
    ValidationScore,
)


def test_cli_parser_accepts_required_paths_and_triton_url():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "input.jpg",
            "output.jpg",
            "--triton-url",
            "localhost:8000",
            "--dict-path",
            "dict.txt",
            "--model-io-path",
            "MODEL_IO.json",
        ]
    )

    assert args.input == Path("input.jpg")
    assert args.output == Path("output.jpg")
    assert args.triton_url == "localhost:8000"
    assert args.dict_path == Path("dict.txt")
    assert args.model_io_path == Path("MODEL_IO.json")


def test_cli_parser_accepts_runtime_tuning_overrides():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "input.jpg",
            "output.jpg",
            "--dict-path",
            "dict.txt",
            "--min-ensemble-score",
            "0.72",
            "--min-score-margin",
            "0.08",
            "--strong-zero-score",
            "0.95",
            "--fine-angle-min-confidence",
            "0.35",
            "--validation-min-score",
            "0.65",
            "--validation-min-margin",
            "0.22",
            "--crops-per-candidate",
            "6",
            "--ocr-max-width",
            "512",
            "--candidate-top-k",
            "3",
            "--candidate-dedupe-degrees",
            "3.5",
        ]
    )

    config = cli.build_config(args)

    assert config.min_ensemble_score == 0.72
    assert config.min_score_margin == 0.08
    assert config.strong_zero_score == 0.95
    assert config.fine_angle_min_confidence == 0.35
    assert config.validation_min_score == 0.65
    assert config.validation_min_margin == 0.22
    assert config.crops_per_candidate == 6
    assert config.ocr_max_width == 512
    assert config.candidate_top_k == 3
    assert config.candidate_dedupe_degrees == 3.5


def test_read_alphabet_uses_paddle_ctc_dictionary_order(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("ga\nna\n", encoding="utf-8")

    assert cli.read_alphabet(path) == ["ga", "na", " ", ""]


def test_read_model_io_loads_tensor_names(tmp_path):
    path = tmp_path / "MODEL_IO.json"
    path.write_text(
        '{"orientation_paddle_doc_ori": {"input": "x", "output": "fetch_name_0", "input_shape": [1, 3, 224, 224]}}',
        encoding="utf-8",
    )

    result = cli.read_model_io(path)

    assert result["orientation_paddle_doc_ori"].input == "x"
    assert result["orientation_paddle_doc_ori"].output == "fetch_name_0"
    assert result["orientation_paddle_doc_ori"].input_shape == [1, 3, 224, 224]


def test_format_trace_lines_includes_model_candidates_validation_and_final_angle(tmp_path):
    result = RotationResult(
        input_path=tmp_path / "input.png",
        output_path=tmp_path / "output.png",
        decision=RotationDecision(angle=90.0, should_rotate=True, reason="recognition_best"),
        elapsed_ms=12.5,
        trace=RotationTrace(
            model_scores=[
                OrientationScores("deep_image", [AngleScore(90.0, 0.9), AngleScore(270.0, 0.1)]),
            ],
            candidate_angles=[AngleCandidate(90.0, "deep_image:90"), AngleCandidate(270.0, "inverse:90")],
            validation_scores=[
                ValidationScore(90.0, 2.1, 0.92, 0.70, 18, 0.0),
                ValidationScore(270.0, 1.1, 0.50, 0.20, 4, 0.2),
            ],
        ),
    )

    output = "\n".join(cli.format_trace_lines(result))

    assert "model=deep_image" in output
    assert "candidates=90.00,270.00" in output
    assert "angle=90.00 score=2.100" in output
    assert "final angle=90.00 rotate=True reason=recognition_best" in output
