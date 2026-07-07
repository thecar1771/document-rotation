from pathlib import Path

from medical_doc_rotation.cli import build_parser, read_alphabet, read_model_io


def test_cli_parser_accepts_required_paths_and_triton_url():
    parser = build_parser()

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


def test_read_alphabet_adds_blank_token(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("가\n나\n", encoding="utf-8")

    assert read_alphabet(path) == ["", "가", "나"]


def test_read_model_io_loads_tensor_names(tmp_path):
    path = tmp_path / "MODEL_IO.json"
    path.write_text(
        '{"orientation_paddle_doc_ori": {"input": "x", "output": "fetch_name_0", "input_shape": [1, 3, 224, 224]}}',
        encoding="utf-8",
    )

    result = read_model_io(path)

    assert result["orientation_paddle_doc_ori"].input == "x"
    assert result["orientation_paddle_doc_ori"].output == "fetch_name_0"
    assert result["orientation_paddle_doc_ori"].input_shape == [1, 3, 224, 224]
