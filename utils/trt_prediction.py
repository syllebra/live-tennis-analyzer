"""TensorRT inference helpers.

This module intentionally uses PyTorch's CUDA allocator/streams instead of
PyCUDA.  PyCUDA needs the CUDA toolkit headers and a compiler at install time,
which is exactly the kind of dependency that fails on a fresh Windows/WSL
machine.  PyTorch is already required by the project and can allocate device
memory and create streams out of the box.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import tensorrt as trt


def _torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    try:
        return mapping[np.dtype(dtype)]
    except KeyError as exc:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}") from exc


class _TRTEngine:
    """Minimal TensorRT 10 runner with batch-size-1 static bindings."""

    def __init__(self, engine_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available to PyTorch.  Install a CUDA build of "
                "PyTorch and check `nvidia-smi` inside WSL."
            )
        torch.cuda.init()
        self.engine_path = str(engine_path)
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()
        self.load(self.engine_path)

        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        self.allocate_memory()

    def load(self, engine_path: str) -> None:
        print(f"Loading TensorRT engine from {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as handle:
            engine_data = handle.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {engine_path}\n"
                "The file may be corrupt or built for a different TensorRT version."
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

    def _resolved_shape(self, name: str) -> Tuple[int, ...]:
        shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        # Engines produced by tools/setup_models.py are static.  This fallback
        # keeps dynamic-batch engines usable by running them at batch size 1.
        return tuple(1 if dim < 0 else dim for dim in shape)

    def allocate_memory(self) -> None:
        self.bindings: Dict[str, Dict[str, object]] = {}

        # Inputs first: setting their concrete shapes lets TensorRT resolve
        # output shapes for dynamic engines.
        for name in self.input_names:
            shape = self._resolved_shape(name)
            self.context.set_input_shape(name, shape)

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = self._resolved_shape(name)
            if name in self.input_names:
                self.context.set_input_shape(name, shape)

            dtype = _torch_dtype_from_numpy(trt.nptype(self.engine.get_tensor_dtype(name)))
            device_tensor = torch.empty(shape, dtype=dtype, device=self.device)
            host_tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            self.context.set_tensor_address(name, int(device_tensor.data_ptr()))
            self.bindings[name] = {
                "shape": shape,
                "dtype": dtype,
                "device": device_tensor,
                "host": host_tensor,
            }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise KeyError(f"missing TensorRT input(s): {missing}; expected {self.input_names}")

        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for name in self.input_names:
                binding = self.bindings[name]
                array = np.ascontiguousarray(inputs[name])
                source = torch.as_tensor(array)
                target = binding["device"]
                if tuple(source.shape) != tuple(target.shape):
                    if source.numel() != target.numel():
                        raise ValueError(
                            f"input {name!r} has {source.numel()} elements but engine expects "
                            f"shape {tuple(target.shape)}"
                        )
                    source = source.reshape(target.shape)
                target.copy_(source.to(dtype=target.dtype, device=self.device), non_blocking=True)

            self.context.execute_async_v3(self.stream.cuda_stream)

            for name in self.output_names:
                binding = self.bindings[name]
                binding["host"].copy_(binding["device"], non_blocking=True)

        self.stream.synchronize()
        # Return copies: the pinned host buffers are reused on the next call.
        return {name: self.bindings[name]["host"].numpy().copy() for name in self.output_names}


class TRTInferenceObject(_TRTEngine):
    """RT-DETRv2 engine: inputs ``images``/``orig_target_sizes`` -> dict outputs."""

    def predict(self, blob: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.run(blob)


class TRTInferenceKeypoint(_TRTEngine):
    """Tennis court keypoint engine: 1x3x224x224 -> 28 coordinates."""

    def __init__(self, engine_path, num_keypoints: int, target_dtype=np.float32):
        self.num_keypoints = num_keypoints
        self.target_dtype = target_dtype
        super().__init__(engine_path)

    def predict(self, batch) -> np.ndarray:
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        array = np.asarray(batch, dtype=self.target_dtype)
        name = self.input_names[0]
        output = self.run({name: array})[self.output_names[0]]
        return output.reshape(-1)
