"""
qcls_wasm_accuracy_sim.py

Standalone numerical simulation to assess the ACCURACY and TEST-RETEST
RELIABILITY of a single implementation of the qCLS adaptive
categorical-loudness-scaling procedure (Shen, Petersen & Neely 2024
JASA; Shen, Petersen & Neely 2026 Ear & Hearing) -- here specifically
the C/WebAssembly web-app implementation, run headlessly via Node.js.

This does NOT compare against a second implementation. It only asks:
"given known ground-truth listener profiles, how close does the WASM
implementation's fitted output come to the truth, and how repeatable
is that output across independent simulated runs?"

Ground truth
------------
The MCPF coefficients, false-alarm rates, and ten-boundary ground-truth
profiles are loaded from ``sim/mcpf_parameters.csv``. The MCPF response
model is implemented separately in ``sim/mcpf_simulator.py``.

Metrics (mirrors the validation logic used in the published paper)
--------------------------------------------------------------------
1. Accuracy   : per-listener MAE between WASM-fitted boundaries and
                ground truth, evaluated at the 10 canonical
                frequencies (and optionally at arbitrary probe
                frequencies via interpolation of ground truth).
2. Reliability: per-listener MAD between two (or more) independent
                simulated runs of the WASM implementation against the
                SAME virtual listener (fresh categorical response draws each run).
3. Bias / spread: per-boundary, per-frequency signed error (fitted -
                truth), so systematic over/under-estimation patterns
                (e.g., at edge frequencies or extreme CU categories)
                can be diagnosed, not just averaged away.
4. Convergence  : accuracy as a function of n_trials, useful for
                choosing the shortest test that still hits your
                accuracy target.

Usage
-----
    # Sanity-check the harness with the self-contained dummy adapter:
    python qcls_wasm_accuracy_sim.py --impl dummy \
        --n_trials 60 --n_reps 20 --out_dir ./results

    # Run one of the webapp's phase-2 modes against the local WASM module:
    python sim/qcls_wasm_accuracy_sim.py --impl wasm --mode bayesian \
        --n_trials 60 --n_reps 20 --out_dir ./results

    # Convergence sweep (accuracy vs. number of trials):
    python qcls_wasm_accuracy_sim.py --impl wasm \
        --runner ./sim/wasm_runner.py --wasm_module ./qcls_core.js \
        --sweep_trials 20 40 60 80 100 --n_reps 15 --out_dir ./results

Author: simulation scaffold for shenyi@uw.edu
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CU_STEP = 5.0                 # category unit spacing (0..50 in steps of 5)
if __package__:
    from .mcpf_simulator import FREQS_HZ, N_BOUNDARIES, VirtualListener, load_ground_truth
else:
    from mcpf_simulator import FREQS_HZ, N_BOUNDARIES, VirtualListener, load_ground_truth

MODE_CODES = {"random": 0, "bayesian": 1, "isophon": 2}


# --------------------------------------------------------------------------
# 3. WASM implementation adapter
# --------------------------------------------------------------------------

@dataclass
class SessionResult:
    stimuli: list[tuple[float, float]]      # (freq_hz, level_db) per trial
    responses: list[int]                    # category per trial
    fitted_boundaries: np.ndarray           # (10, 10) fitted profile at FREQS_HZ


class ImplementationAdapter:
    """Common interface an implementation under test must satisfy."""

    name: str = "base"

    def start_session(self, seed: int, n_trials: int) -> None:
        raise NotImplementedError

    def next_stimulus(self) -> tuple[float, float]:
        raise NotImplementedError

    def submit_response(self, category: int) -> None:
        raise NotImplementedError

    def get_fit(self) -> np.ndarray:
        """Return fitted (10 boundaries x 10 canonical freqs) matrix."""
        raise NotImplementedError

    def evaluate_fit(self) -> np.ndarray:
        """Evaluate the current trial history without advancing the session."""
        return self.get_fit()

    def prepare_evaluation(self) -> None:
        """Refresh the implementation estimate at the current trial count."""

    def run_session(self, listener: VirtualListener, n_trials: int, seed: int) -> SessionResult:
        self.start_session(seed, n_trials)
        stimuli, responses = [], []
        for _ in range(n_trials):
            freq_hz, level_db = self.next_stimulus()
            cat = listener.respond(freq_hz, level_db)
            self.submit_response(cat)
            stimuli.append((freq_hz, level_db))
            responses.append(cat)
        self.prepare_evaluation()
        fit = self.get_fit()
        return SessionResult(stimuli=stimuli, responses=responses, fitted_boundaries=fit)

    def run_session_at_points(
        self,
        listener: VirtualListener,
        evaluation_points: list[int],
        seed: int,
    ) -> dict[int, np.ndarray]:
        """Run one cumulative session and fit at each requested trial count."""
        self.start_session(seed, max(evaluation_points))
        fits = {}
        points = set(evaluation_points)
        for trial in range(1, max(evaluation_points) + 1):
            freq_hz, level_db = self.next_stimulus()
            category = listener.respond(freq_hz, level_db)
            self.submit_response(category)
            if trial in points:
                self.prepare_evaluation()
                fits[trial] = self.evaluate_fit()
        return fits


class WasmAdapter(ImplementationAdapter):
    """
    Drives the compiled WASM qCLS module headlessly through the Python
    webapp-session runner and its persistent Node.js Emscripten bridge.
    The runner exposes a JSON-line protocol over
    stdin/stdout, e.g.:

        -> {"cmd": "start", "seed": 12345, "mode": 1, "n_trials": 60}
        <- {"ok": true}
        -> {"cmd": "next_stimulus"}
        <- {"freq_hz": 1000.0, "level_db": 65.0}
        -> {"cmd": "submit_response", "category": 6}
        <- {"ok": true}
        ... (repeat next_stimulus/submit_response for n_trials) ...
        -> {"cmd": "get_fit"}
        <- {"boundaries": [[...10 freqs...], ... x10 rows...]}

    Mode values match the webapp select: 0=Random, 1=Bayesian, 2=isoPhon.
    """

    name = "wasm"

    def __init__(self, runner: str | Path, wasm_module: str | Path, mode: str):
        self.runner = str(runner)
        self.wasm_module = wasm_module
        self.mode = mode
        self.proc: Optional[subprocess.Popen] = None

    def _ensure_proc(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                [sys.executable, self.runner, "--module", str(self.wasm_module)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )

    def _send(self, msg: dict) -> dict:
        self._ensure_proc()
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"wasm_runner produced no output. stderr:\n{err}")
        return json.loads(line)

    def start_session(self, seed: int, n_trials: int) -> None:
        resp = self._send({
            "cmd": "start", "seed": int(seed), "mode": MODE_CODES[self.mode],
            "n_trials": int(n_trials),
        })
        if not resp.get("ok", False):
            raise RuntimeError(f"WASM start_session failed: {resp}")

    def next_stimulus(self) -> tuple[float, float]:
        resp = self._send({"cmd": "next_stimulus"})
        return float(resp["freq_hz"]), float(resp["level_db"])

    def submit_response(self, category: int) -> None:
        resp = self._send({"cmd": "submit_response", "category": int(category)})
        if not resp.get("ok", False):
            raise RuntimeError(f"WASM submit_response failed: {resp}")

    def get_fit(self) -> np.ndarray:
        return self._get_fit_command("get_fit")

    def evaluate_fit(self) -> np.ndarray:
        return self._get_fit_command("evaluate_fit")

    def prepare_evaluation(self) -> None:
        # The Python runner updates the C tracker with every submitted response.
        pass

    def _get_fit_command(self, command: str) -> np.ndarray:
        resp = self._send({"cmd": command})
        mat = np.array(resp["boundaries"], dtype=float)
        if mat.shape != (N_BOUNDARIES, len(FREQS_HZ)):
            raise ValueError(f"WASM get_fit returned shape {mat.shape}, expected (10, 10). "
                              f"Ensure wasm_runner.py returns the canonical "
                              f"frequencies {FREQS_HZ.tolist()} before returning.")
        return mat

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.terminate()


class DummyAdapter(ImplementationAdapter):
    """
    Self-contained reference implementation for harness sanity checks.
    Implements a simple adaptive rule: for each frequency in turn,
    home in on each boundary using a shrinking-step-size bisection
    driven by the "> vs <=" category comparison from the listener's
    response, then linearly resample to the 10 canonical frequencies.
    NOT meant to model your real WASM procedure -- just a stand-in so
    the surrounding metrics/plumbing can be tested without Node/WASM.
    """

    name = "dummy"

    def __init__(self):
        self.rng = None
        self._trial_freqs = np.repeat(FREQS_HZ, 1)
        self._history: dict[float, list[tuple[float, int]]] = {}

    def start_session(self, seed: int, n_trials: int) -> None:
        self.rng = np.random.default_rng(seed)
        self._freq_cycle_idx = 0
        self._history = {f: [] for f in FREQS_HZ}
        self._current_level = {f: 50.0 for f in FREQS_HZ}  # start mid-range
        self._step = {f: 20.0 for f in FREQS_HZ}
        self._last_freq = None

    def next_stimulus(self) -> tuple[float, float]:
        f = FREQS_HZ[self._freq_cycle_idx % len(FREQS_HZ)]
        self._freq_cycle_idx += 1
        self._last_freq = f
        level = float(np.clip(self._current_level[f], 0, 110))
        return float(f), level

    def submit_response(self, category: int) -> None:
        f = self._last_freq
        self._history[f].append((self._current_level[f], category))
        target_cat = 5  # crude: chase category 5 (mid CU) with shrinking steps
        if category > target_cat:
            self._current_level[f] -= self._step[f]
        elif category < target_cat:
            self._current_level[f] += self._step[f]
        self._step[f] *= 0.85

    def get_fit(self) -> np.ndarray:
        # Crude post-hoc fit: for each frequency, rank the tested levels
        # and slot them into 10 boundary "bins" by sorted level.
        fit = np.zeros((N_BOUNDARIES, len(FREQS_HZ)))
        for fi, f in enumerate(FREQS_HZ):
            levels = sorted([lvl for lvl, _ in self._history[f]])
            if len(levels) < N_BOUNDARIES:
                levels = levels + [levels[-1] if levels else 50.0] * (N_BOUNDARIES - len(levels))
            # take evenly spaced order statistics as boundary estimates
            idxs = np.linspace(0, len(levels) - 1, N_BOUNDARIES).astype(int)
            fit[:, fi] = np.array(levels)[idxs]
        fit = np.sort(fit, axis=0)  # enforce monotonic boundaries
        return fit


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------

def mae(fit: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(fit - truth)))


def mad_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def signed_error(fit: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return fit - truth


def plot_profile_comparison(
    listener_id: str,
    repetition: int,
    truth: np.ndarray,
    estimate: np.ndarray,
) -> None:
    """Show truth and fitted boundaries, pausing until a key is pressed."""
    import matplotlib.pyplot as plt

    frequencies_khz = FREQS_HZ / 1000.0
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, N_BOUNDARIES))
    for boundary, color in enumerate(colors):
        category = int((boundary + 1) * CU_STEP)
        axis.plot(
            frequencies_khz, truth[boundary], "--", color=color,
            linewidth=1.5, label=f"CU {category} truth" if boundary == 0 else None,
        )
        axis.plot(
            frequencies_khz, estimate[boundary], "-", color=color,
            linewidth=1.5, label=f"CU {category} estimate" if boundary == 0 else None,
        )

    axis.set_xscale("log")
    axis.set_xticks(frequencies_khz)
    axis.set_xticklabels([f"{frequency:g}" for frequency in frequencies_khz])
    axis.set_xlabel("Frequency (kHz)")
    axis.set_ylabel("Level (dB SPL)")
    axis.set_title(f"Listener {listener_id}, repetition {repetition}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    plt.show(block=False)
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    print("  Plot displayed; press any key in the figure window to continue.", flush=True)
    figure.waitforbuttonpress()
    plt.close(figure)


# --------------------------------------------------------------------------
# 5. Core experiment runner
# --------------------------------------------------------------------------

def run_accuracy_reliability(
    listeners,
    adapter_factory,
    n_trials: int,
    n_reps: int,
    base_seed: int = 0,
    plot_repetitions: bool = False,
    mode: str = "bayesian",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    """
    For each listener, run `n_reps` independent simulated sessions of
    the WASM implementation (fresh categorical response draws + fresh RNG
    seed each time). Compute:
      - accuracy per rep (MAE vs. ground truth)
      - pairwise test-retest MAD across all rep pairs for that listener

    Returns
    -------
    accuracy_df : one row per (listener, rep) with MAE
    reliability_df : one row per (listener, rep_pair) with MAD
    signed_error_records : (returned separately, see caller) per-cell
                            signed error for bias diagnostics
    """
    acc_rows = []
    fits_by_listener: dict[str, list[np.ndarray]] = {}

    for lid, gt in listeners.items():
        fits = []
        print(f"Processing listener {lid}", flush=True)
        adapter = adapter_factory()
        try:
            for r in range(n_reps):
                print(f"  rep {r + 1}/{n_reps}...", end="\n", flush=True)
                seed = base_seed + hash((lid, r)) % 10_000_000
                rng = np.random.default_rng(seed)
                listener = listeners.new_virtual_listener(lid, rng)
                result = adapter.run_session(listener, n_trials=n_trials, seed=seed)
                if plot_repetitions:
                    plot_profile_comparison(
                        listener_id=lid,
                        repetition=r + 1,
                        truth=gt,
                        estimate=result.fitted_boundaries,
                    )
                listener_mae = mae(result.fitted_boundaries, gt)
                acc_rows.append({
                    "mode": mode, "listener_id": lid, "rep": r, "mae_db": listener_mae
                })
                fits.append(result.fitted_boundaries)
        finally:
            if hasattr(adapter, "close"):
                adapter.close()
        fits_by_listener[lid] = fits

    accuracy_df = pd.DataFrame(acc_rows)

    rel_rows = []
    for lid, fits in fits_by_listener.items():
        n = len(fits)
        for i in range(n):
            for j in range(i + 1, n):
                rel_rows.append({
                    "mode": mode, "listener_id": lid, "rep_i": i, "rep_j": j,
                    "mad_db": mad_between(fits[i], fits[j]),
                })
    reliability_df = pd.DataFrame(rel_rows)

    return accuracy_df, reliability_df, fits_by_listener


def bias_by_cell(
    listeners,
    fits_by_listener: dict[str, list[np.ndarray]],
    mode: str = "bayesian",
) -> pd.DataFrame:
    """Signed error (fit - truth) averaged across reps, per listener/boundary/freq."""
    rows = []
    for lid, fits in fits_by_listener.items():
        gt = listeners[lid]
        stacked = np.stack(fits, axis=0)          # (n_reps, 10, 10)
        mean_signed = np.mean(stacked - gt[None], axis=0)  # (10, 10)
        for b in range(N_BOUNDARIES):
            for fi, f in enumerate(FREQS_HZ):
                rows.append({
                    "mode": mode, "listener_id": lid, "boundary_index": b + 1,
                    "freq_hz": f, "signed_bias_db": mean_signed[b, fi],
                })
    return pd.DataFrame(rows)


def convergence_sweep(
    listeners,
    adapter_factory,
    trial_counts: list[int],
    n_reps: int,
    base_seed: int = 0,
    plot_repetitions: bool = False,
    mode: str = "bayesian",
) -> pd.DataFrame:
    """Accuracy (population-mean MAE) as a function of n_trials."""
    rows = []
    max_trials = max(trial_counts)
    errors_by_point = {nt: [] for nt in trial_counts}
    for lid, gt in listeners.items():
        adapter = adapter_factory()
        print(f"Processing listener {lid}", flush=True)
        try:
            for repetition in range(n_reps):
                print(f"  rep {repetition + 1}/{n_reps}...", end="\n", flush=True)
                seed = base_seed + hash((lid, repetition)) % 10_000_000
                listener = listeners.new_virtual_listener(
                    lid, np.random.default_rng(seed)
                )
                fits = adapter.run_session_at_points(listener, trial_counts, seed)
                for nt, fit in fits.items():
                    errors_by_point[nt].append(mae(fit, gt))
                    if plot_repetitions:
                        plot_profile_comparison(lid, nt, gt, fit)
        finally:
            if hasattr(adapter, "close"):
                adapter.close()

    for nt in trial_counts:
        errors = pd.Series(errors_by_point[nt])
        rows.append({
            "mode": mode, "n_trials": nt,
            "mean_mae_db": errors.mean(),
            "sd_mae_db": errors.std(),
            "median_mae_db": errors.median(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------

def build_adapter_factory(args):
    if args.impl == "dummy":
        if args.mode != "bayesian":
            raise ValueError("--mode is only applied by --impl wasm; the dummy adapter has no webapp modes")
        return lambda: DummyAdapter()
    elif args.impl == "wasm":
        if not args.runner or not args.wasm_module:
            raise ValueError("--impl wasm requires --runner and --wasm_module")
        return lambda: WasmAdapter(args.runner, args.wasm_module, args.mode)
    else:
        raise ValueError(f"Unknown --impl {args.impl}")


def main():
    p = argparse.ArgumentParser(description="Assess WASM qCLS implementation accuracy & reliability.")
    p.add_argument(
        "--ground_truth_csv", type=Path,
        default=Path(__file__).resolve().with_name("mcpf_parameters.csv"),
        help="CSV containing MCPF parameters and ground-truth profiles",
    )
    p.add_argument("--impl", choices=["wasm", "dummy"], default="wasm")
    p.add_argument(
        "--mode", choices=MODE_CODES, default="bayesian",
        help="Webapp phase-2 mode: random, bayesian, or isophon",
    )
    p.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name("wasm_runner.py"),
        help="Path to the Python webapp-session runner",
    )
    p.add_argument(
        "--wasm_module", type=Path,
        default=Path(__file__).resolve().parents[1] / "qcls_core.js",
        help="Path to the Emscripten JavaScript module",
    )
    p.add_argument("--n_trials", type=int, default=60, help="Trials per session (ignored if --sweep_trials given)")
    p.add_argument("--n_reps", type=int, default=20, help="Independent repeated sessions per listener")
    p.add_argument("--sweep_trials", type=int, nargs="+", default=None,
                   help="If given, run a convergence sweep over these trial counts instead of a single run")
    p.add_argument("--plot_repetitions", action="store_true",
                   help="Plot truth and fitted boundaries after each repetition; press a key to continue")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--out_dir", default="./wasm_accuracy_results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    listeners = load_ground_truth(args.ground_truth_csv)
    print(f"Loaded {len(listeners)} listener(s) from {args.ground_truth_csv}")

    adapter_factory = build_adapter_factory(args)

    if args.sweep_trials:
        print(f"Running convergence sweep over trial counts: {args.sweep_trials}")
        sweep_df = convergence_sweep(
            listeners, adapter_factory, trial_counts=args.sweep_trials,
            n_reps=args.n_reps, base_seed=args.base_seed,
            plot_repetitions=args.plot_repetitions,
            mode=args.mode,
        )
        sweep_df.to_csv(out_dir / "convergence_sweep.csv", index=False)
        print(sweep_df.to_string(index=False))
        print(f"\nSaved: {out_dir / 'convergence_sweep.csv'}")
        return

    print(f"Running {args.n_reps} reps x {len(listeners)} listeners "
                    f"({args.n_trials} trials/session, mode={args.mode}, MCPF response simulation)...")
    accuracy_df, reliability_df, fits_by_listener = run_accuracy_reliability(
        listeners, adapter_factory, n_trials=args.n_trials, n_reps=args.n_reps,
        base_seed=args.base_seed, plot_repetitions=args.plot_repetitions,
        mode=args.mode,
    )
    bias_df = bias_by_cell(listeners, fits_by_listener, mode=args.mode)

    accuracy_df.to_csv(out_dir / "accuracy_per_rep.csv", index=False)
    reliability_df.to_csv(out_dir / "reliability_per_pair.csv", index=False)
    bias_df.to_csv(out_dir / "signed_bias_by_cell.csv", index=False)

    summary = {
        "mode": args.mode,
        "n_listeners": len(listeners),
        "n_trials": args.n_trials,
        "n_reps": args.n_reps,
        "accuracy_mean_mae_db": accuracy_df["mae_db"].mean(),
        "accuracy_sd_mae_db": accuracy_df["mae_db"].std(),
        "accuracy_median_mae_db": accuracy_df["mae_db"].median(),
        "accuracy_max_mae_db": accuracy_df["mae_db"].max(),
        "reliability_mean_mad_db": reliability_df["mad_db"].mean(),
        "reliability_sd_mad_db": reliability_df["mad_db"].std(),
        "reliability_median_mad_db": reliability_df["mad_db"].median(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    per_listener = accuracy_df.groupby("listener_id")["mae_db"].mean().sort_values(ascending=False)
    print("\n=== Per-listener mean accuracy (worst first) ===")
    print(per_listener.to_string())

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
