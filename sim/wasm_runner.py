"""JSON-lines session runner mirroring the qCLS webapp's adaptive logic."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

BANDS_HZ = (250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000)
MODE_CODES = {"random": 0, "bayesian": 1, "isophon": 2}
MAX_OUTPUT_SPL = 110.0
DISCOMFORT_BACKOFF_DB = 5.0
BRIDGE_PATH = Path(__file__).with_name("wasm_bridge.js")


class WasmBridge:
    """Synchronous JSON client for the persistent Emscripten Node bridge."""

    def __init__(self, module_path: Path, node: str):
        self.process = subprocess.Popen(
            [node, str(BRIDGE_PATH), "--module", str(module_path.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"WASM bridge exited unexpectedly: {error}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("WASM bridge pipes are unavailable")

        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"WASM bridge returned no output: {error}")
        response = json.loads(line)
        if response.get("ok") is False:
            raise RuntimeError(response.get("error", "WASM bridge request failed"))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)


class WebAppSession:
    """Default qCLS webapp session state without browser/audio concerns."""

    def __init__(self, bridge: WasmBridge):
        self.bridge = bridge
        self.history_f: list[float] = []
        self.history_l: list[float] = []
        self.history_r: list[float] = []

    def start(self, seed: int, mode: int, total_trials: int) -> None:
        if mode not in MODE_CODES.values():
            raise ValueError(f"Unknown webapp mode code: {mode}")
        if total_trials < 1:
            raise ValueError("Session trial count must be positive")

        self.mode = mode
        self.total_trials = total_trials
        self.rng = random.Random(seed)
        self.start_level = 50.0
        self.current_frequency = 1000.0
        self.current_level = self.start_level
        self.min_level = -10.0
        self.ceiling = 100.0
        self.max_level = self.ceiling
        self.phase = "P1"
        self.phase1_direction = -1
        self.start_response: int | None = None
        self.history_f = []
        self.history_l = []
        self.history_r = []
        self.bridge.request({"cmd": "init"})

    def next_stimulus(self) -> tuple[float, float]:
        return self.current_frequency, self.current_level

    def submit_response(self, category: int) -> None:
        if category < 0 or category > 10:
            raise ValueError(f"Virtual listener category must be in 0..10, got {category}")

        response_cu = int(category) * 5
        self.bridge.request({
            "cmd": "update",
            "frequency": self.current_frequency,
            "level": self.current_level,
            "response": response_cu,
        })
        self.history_f.append(self.current_frequency)
        self.history_l.append(self.current_level)
        self.history_r.append(float(response_cu))
        if len(self.history_f) < self.total_trials:
            self._update_adaptive_logic(response_cu)

    def _backoff_from(self, level: float) -> float:
        return max(self.min_level + 10.0, level - DISCOMFORT_BACKOFF_DB)

    def _select_phase2(self) -> None:
        if self.mode == MODE_CODES["random"]:
            self.current_frequency = float(self.rng.choice(BANDS_HZ))
            if self.rng.random() > 0.15:
                level_range = max(0.0, self.max_level - self.min_level - 1.0)
                self.current_level = math.floor(
                    self.min_level + self.rng.random() * level_range + 0.5
                )
            else:
                self.current_level = math.floor(
                    self.min_level - self.rng.random() * 10.0 + 0.5
                )
        else:
            command = "bayesian" if self.mode == MODE_CODES["bayesian"] else "isophon"
            stimulus = self.bridge.request({
                "cmd": command,
                "min_level": self.min_level,
                "max_level": self.max_level,
            })
            self.current_frequency = float(stimulus["frequency"])
            self.current_level = float(stimulus["level"])

    def _update_adaptive_logic(self, response_cu: int) -> None:
        if self.phase == "P1":
            if self.start_response is None:
                self.start_response = response_cu

            if self.phase1_direction == -1:
                if response_cu == 0 or self.current_level <= 0:
                    self.min_level = self.current_level
                    self.phase1_direction = 1
                    if self.start_response == 50:
                        self.max_level = self._backoff_from(self.start_level)
                        self.phase = "P2"
                        self._select_phase2()
                    else:
                        step = 5.0 if self.start_response >= 35 else 10.0
                        self.current_level = self.start_level + step
                else:
                    self.current_level -= 10.0
            elif response_cu == 50 or self.current_level >= self.ceiling:
                self.max_level = self._backoff_from(self.current_level)
                self.phase = "P2"
                self._select_phase2()
            else:
                step = 5.0 if self.start_response >= 35 else 10.0
                self.current_level += step
        else:
            if response_cu == 50:
                self.max_level = min(self.max_level, self._backoff_from(self.current_level))
            if response_cu == 0 and self.current_level > self.min_level:
                self.min_level = self.current_level
            self._select_phase2()

        if self.current_level > self.ceiling:
            self.current_level = self.ceiling
        if self.current_level > self.max_level:
            self.current_level = self.max_level
        if self.current_level < -10.0:
            self.current_level = -10.0

    def fitted_boundaries(self) -> list[list[float]]:
        fit = self.bridge.request({
            "cmd": "fit",
            "frequencies": self.history_f,
            "levels": self.history_l,
            "responses": self.history_r,
        })["boundaries"]
        return [
            [float(fit[frequency * 10 + boundary]) for frequency in range(10)]
            for boundary in range(10)
        ]


def serve(module_path: Path, node: str) -> int:
    try:
        bridge = WasmBridge(module_path, node)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        return 1

    session = WebAppSession(bridge)
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                command = message.get("cmd")
                if command == "start":
                    session.start(
                        int(message["seed"]), int(message["mode"]), int(message["n_trials"])
                    )
                    response = {"ok": True}
                elif command == "next_stimulus":
                    frequency, level = session.next_stimulus()
                    response = {"freq_hz": frequency, "level_db": level}
                elif command == "submit_response":
                    session.submit_response(int(message["category"]))
                    response = {"ok": True}
                elif command in ("get_fit", "evaluate_fit"):
                    response = {"boundaries": session.fitted_boundaries()}
                else:
                    raise ValueError(f"Unknown runner command: {command}")
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            print(json.dumps(response), flush=True)
    finally:
        bridge.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True, help="Path to qcls_core.js")
    parser.add_argument("--node", default="node", help="Node.js executable")
    args = parser.parse_args()
    return serve(args.module, args.node)


if __name__ == "__main__":
    raise SystemExit(main())