from pathlib import Path

from medical_doc_rotation.cli import build_parser, read_alphabet


def test_cli_parser_accepts_required_paths_and_triton_url():
    parser = build_parser()

    args = parser.parse_args(["input.jpg", "output.jpg", "--triton-url", "localhost:8000", "--dict-path", "dict.txt"])

    assert args.input == Path("input.jpg")
    assert args.output == Path("output.jpg")
    assert args.triton_url == "localhost:8000"
    assert args.dict_path == Path("dict.txt")


def test_read_alphabet_adds_blank_token(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("가\n나\n", encoding="utf-8")

    assert read_alphabet(path) == ["", "가", "나"]
