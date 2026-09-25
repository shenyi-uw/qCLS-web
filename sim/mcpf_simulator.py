"""MCPF-based virtual listener model backed by exported CSV parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FREQS_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000], dtype=float)
N_BOUNDARIES = 10
COEFFICIENT_COLUMNS = [f"coeff_{index:02d}" for index in range(1, 21)]
PROFILE_COLUMNS = [f"cu_{index * 5:02d}" for index in range(1, 11)]


class VirtualListener:
    """Simulate categorical responses from one listener's MCPF parameters."""

    def __init__(
        self,
        listener_id: str,
        parameter_rows: pd.DataFrame,
        rng: Optional[np.random.Generator] = None,
    ):
        self.listener_id = listener_id
        self.rng = rng if rng is not None else np.random.default_rng()
        rows = parameter_rows.sort_values("freq_hz")
        self.freqs_hz = rows["freq_hz"].to_numpy(dtype=float)
        if not np.array_equal(self.freqs_hz, FREQS_HZ):
            raise ValueError(
                f"Listener {listener_id} must have exactly the canonical frequencies "
                f"{FREQS_HZ.tolist()}"
            )

        coefficients = rows[COEFFICIENT_COLUMNS].to_numpy(dtype=float).T
        false_alarm = rows["false_alarm_rate"].to_numpy(dtype=float)
        self.levels = np.arange(0.0, 120.0001, 0.1)
        self.pf = self._make_mcpf(coefficients, false_alarm)
        self.gt_matrix = rows[PROFILE_COLUMNS].to_numpy(dtype=float).T

    def _make_mcpf(self, coefficients: np.ndarray, false_alarm: np.ndarray) -> np.ndarray:
        """Translate the MATLAB make_mcpf function, shape (freq, level, category)."""
        nfreqs = len(self.freqs_hz)
        pf = np.zeros((nfreqs, len(self.levels), N_BOUNDARIES))
        dfa = false_alarm / 11.0
        midpoint = np.cumsum(coefficients[10:20], axis=0)

        for frequency in range(nfreqs):
            for boundary in range(N_BOUNDARIES):
                coefficient = coefficients[boundary, frequency]
                intercept = -midpoint[boundary, frequency] * coefficient
                logit = np.clip(intercept + coefficient * self.levels, -700.0, 700.0)
                logistic = 1.0 / (1.0 + np.exp(-logit))
                pf[frequency, :, boundary] = (
                    logistic * (1.0 - false_alarm[frequency]) + boundary * dfa[frequency]
                )

        for _ in range(20):
            differences = np.diff(pf, axis=2)
            for boundary in range(N_BOUNDARIES - 1):
                mask = differences[:, :, boundary] < (dfa[:, None] / 2.0)
                adjustment = differences[:, :, boundary] / 2.0 - dfa[:, None] / 2.0
                pf[:, :, boundary] = np.where(
                    mask, pf[:, :, boundary] + adjustment, pf[:, :, boundary]
                )
                pf[:, :, boundary + 1] = np.where(
                    mask, pf[:, :, boundary] + dfa[:, None], pf[:, :, boundary + 1]
                )

        upper = (1.0 - dfa)[:, None]
        for boundary in range(N_BOUNDARIES - 1, -1, -1):
            mask = pf[:, :, boundary] > upper
            pf[:, :, boundary] = np.where(mask, upper, pf[:, :, boundary])
            upper = pf[:, :, boundary] - dfa[:, None]

        lower = dfa[:, None]
        for boundary in range(N_BOUNDARIES):
            mask = pf[:, :, boundary] < lower
            pf[:, :, boundary] = np.where(mask, lower, pf[:, :, boundary])
            lower = pf[:, :, boundary] + dfa[:, None]
        return pf

    def respond(self, freq_hz: float, level_db: float) -> int:
        """Return a simulated category response (0-10) for one trial."""
        frequency_position = np.interp(
            np.log2(freq_hz), np.log2(self.freqs_hz), np.arange(len(self.freqs_hz))
        )
        low = int(np.floor(frequency_position))
        high = min(low + 1, len(self.freqs_hz) - 1)
        fraction = frequency_position - low
        level_index = int(np.argmin(np.abs(self.levels - level_db)))
        cdf = (
            (1.0 - fraction) * self.pf[low, level_index, :]
            + fraction * self.pf[high, level_index, :]
        )
        random_value = self.rng.random()
        matlab_category = (
            11
            if random_value > cdf[-1]
            else int(np.searchsorted(cdf, random_value, side="right")) + 1
        )
        return matlab_category - 1

    def full_profile(self) -> np.ndarray:
        """Ground-truth boundaries at the ten canonical frequencies."""
        return self.gt_matrix.copy()


class GroundTruthProfiles:
    """Cached CSV-backed listener models and their boundary profiles."""

    def __init__(self, parameter_csv: str | Path):
        parameter_csv = Path(parameter_csv)
        parameters = pd.read_csv(parameter_csv)
        required_columns = {
            "listener_id", "freq_hz", "false_alarm_rate",
            *COEFFICIENT_COLUMNS, *PROFILE_COLUMNS,
        }
        missing_columns = required_columns.difference(parameters.columns)
        if missing_columns:
            raise ValueError(
                f"{parameter_csv} is missing required columns: {sorted(missing_columns)}"
            )
        if parameters.duplicated(["listener_id", "freq_hz"]).any():
            raise ValueError(f"{parameter_csv} has duplicate listener/frequency rows")

        self._models: dict[str, VirtualListener] = {}
        for listener_id, rows in parameters.groupby("listener_id", sort=True):
            listener_key = str(int(listener_id))
            if len(rows) != len(FREQS_HZ):
                raise ValueError(
                    f"Listener {listener_key} has {len(rows)} rows; expected {len(FREQS_HZ)}"
                )
            self._models[listener_key] = VirtualListener(listener_key, rows)

        if not self._models:
            raise ValueError(f"No listener parameters found in {parameter_csv}")

    def __getitem__(self, listener_id: str) -> np.ndarray:
        return self._models[listener_id].full_profile()

    def __len__(self) -> int:
        return len(self._models)

    def items(self):
        return ((listener_id, model.full_profile()) for listener_id, model in self._models.items())

    def new_virtual_listener(
        self, listener_id: str, rng: np.random.Generator
    ) -> VirtualListener:
        listener = self._models[listener_id]
        listener.rng = rng
        return listener


def load_ground_truth(parameter_csv: str | Path) -> GroundTruthProfiles:
    """Load listener response parameters and ground-truth profiles from CSV."""
    return GroundTruthProfiles(parameter_csv)