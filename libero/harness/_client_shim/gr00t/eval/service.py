# Client-side stand-in for Isaac-GR00T gr00t/eval/service.py (https://github.com/NVIDIA/Isaac-GR00T).
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-ZMQ stand-in for gr00t.eval.service on the LIBERO *client* side.

The GR00T harness does `from gr00t.eval.service import
ExternalRobotInferenceClient`. The client runs in .venv-libero (robosuite 1.4.0
+ official LIBERO), which deliberately does NOT install the heavy gr00t package.
The wire format (msgpack + __ndarray_class__/np.save) is byte-identical to the
gr00t server's MsgSerializer, so we inline a minimal REQ-socket client here.

Fully self-contained: depends only on msgpack/zmq/numpy. No RoboCerebra import.
"""
import io

import msgpack
import numpy as np
import zmq


class _MsgSerializer:
    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)


class ExternalRobotInferenceClient:
    def __init__(self, host="127.0.0.1", port=5555, timeout_ms=120000):
        self.context = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.socket.send(_MsgSerializer.to_bytes(req))
        resp = _MsgSerializer.from_bytes(self.socket.recv())
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_action(self, observations: dict) -> dict:
        resp = self.call_endpoint("get_action", observations)
        if isinstance(resp, (list, tuple)):
            return resp[0]
        return resp
