from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TritonClientProtocol(Protocol):
    def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str],
        timeout_ms: int,
    ) -> dict[str, np.ndarray]:
        ...

    def is_model_ready(self, model_name: str) -> bool:
        ...


@dataclass
class TritonHttpClient:
    url: str

    def __post_init__(self) -> None:
        import tritonclient.http as httpclient

        self._client = httpclient.InferenceServerClient(url=self.url)

    def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str],
        timeout_ms: int,
    ) -> dict[str, np.ndarray]:
        import tritonclient.http as httpclient

        triton_inputs = []
        for name, value in inputs.items():
            request_input = httpclient.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
            request_input.set_data_from_numpy(value)
            triton_inputs.append(request_input)
        triton_outputs = [httpclient.InferRequestedOutput(name) for name in outputs]
        result = self._client.infer(
            model_name,
            triton_inputs,
            outputs=triton_outputs,
            request_timeout=timeout_ms / 1000,
        )
        return {name: result.as_numpy(name) for name in outputs}

    def is_model_ready(self, model_name: str) -> bool:
        return bool(self._client.is_model_ready(model_name))


def np_to_triton_dtype(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    if dtype == np.uint8:
        return "UINT8"
    if dtype == np.int64:
        return "INT64"
    raise ValueError(f"Unsupported Triton dtype: {dtype}")
