# qCLS Web Application: Architectural Overview & User Guide

## 1. Introduction & Technology Stack
The **qCLS** web application is a fully client-side, browser-based tool for measuring categorical loudness growth functions. To achieve real-time, high-performance execution of the Bayesian Adaptive Procedure (BAP) without relying on a remote server, the application utilizes a hybrid architecture:

- **Front-End (HTML/JavaScript):** Handles the user interface, state management, and real-time audio rendering via the Web Audio API.
- **Back-End (C / WebAssembly):** The heavy mathematical lifting—including stimulus generation, psychoacoustic interpolation, and the Extended Kalman Filter—has been translated from objective MATLAB into C. This C code is compiled into WebAssembly (Wasm) using Emscripten, allowing it to run at near-native speeds directly in the user's browser.
- **Security & Privacy:** Because the entire application runs locally in the browser, no participant data (PHI) or trial history is ever transmitted over the internet.

## 2. Translating the Bayesian Tracker (MATLAB to C)
To replicate the MATLAB BAP algorithm in a web environment, the core logic was translated into standard C. Here is how the specific mathematical dependencies were handled:

### A. State Management & Structs
The dynamic qCLS MATLAB class was flattened into static C structs (`qCLS_Config` and `qCLS_State`). Memory footprints are allocated upfront (e.g., matrices for the 9 frequency bands and 11 categories) to ensure memory safety in the browser.

### B. Replicating pchip (Monotonic Interpolation)
MATLAB relies on `pchip` to interpolate categorical boundaries smoothly without overshooting. Because a standard cubic spline would allow boundaries to cross (violating monotonicity), a custom C function (`interpolate_pchip`) was written. This function implements the exact **Fritsch-Carlson shape-preserving interpolant** algorithm used by MATLAB, ensuring the psychometric boundary curves are mathematically identical.

### C. Linear Algebra Engine
The Kalman Filter update (`Kalman_update`) relies heavily on matrix multiplication, transposition, and inversion (e.g., calculating the Kalman gain $K$ and updating the covariance matrix $P$). A custom, lightweight linear algebra engine was built into the C code. Most notably, matrix inversion is handled via **Gauss-Jordan elimination with partial pivoting**, which is highly stable for the 10x10 and 12x12 matrices used in this procedure.

### D. Model Mixture and Posterior-Driven Selection
The engine carries all **21 anchor-frequency models**, matching the MATLAB. It
enumerates `nchoosek(2:8, 2)` with the endpoints fixed at bands 1 and 9, runs a
Kalman update on each per trial, and accumulates a log-likelihood per model.

Results are read as a **mixture** rather than from the single most likely model.
By the law of total variance the reported variance is the within-model term plus
the between-model term, so the disagreement between models becomes part of the
reported uncertainty instead of being discarded.

Stimulus selection consults the posterior: it presents the band and boundary
where the mixture is least certain, at that boundary's own mean. A trial there
lowers that uncertainty and the next maximum moves elsewhere, so the rule
spreads over the run rather than sticking.

> Earlier versions ran one fixed model and selected frequency and level at
> random, which made the interface's "Bayesian" option identical to its
> "Random" one. Both arms now draw from the same nine audiometric band centres,
> so Random is a genuine control for Bayesian rather than a different
> experiment.

## 3. Application Flow & Phase Logic

### Phase 1: Dynamic Range Bounding
Before the BAP takes over, Phase 1 establishes the listener's dynamic range using a rule-based tracking procedure:
1. **First Presentation:** Plays the specified Start Level (default 50 dB SPL) and records the initial response.
2. **Descending:** The level decreases in 10 dB steps until the listener responds "Can't Hear" or the level reaches 0 dB SPL. This locks in the absolute minimum level.
3. **Ascending:** The test jumps back to the initial Start Level. If the very first response was "Loud" or greater, the ascending step size is set to **5 dB**. Otherwise, it uses **10 dB** steps. It ascends until the listener responds "Too Loud" or the level reaches the **Max Output** ceiling. This locks in the working maximum, which thereafter only ratchets downward (see Safety).

### Phase 2: Bayesian Adaptive Tracking
Once bounded, Phase 2 begins. After every response:
1. JavaScript sends the completed trial to the compiled C engine.
2. The C engine updates all 21 candidate anchor-frequency models once with that trial.
3. The selected adaptive mode requests the next frequency and level from the current posterior, bounded by the Phase 1 limits and the Max Output ceiling.

The posterior is updated incrementally. The completed trial is not appended to a
history buffer and replayed for each stimulus decision. Bayesian mode selects
the most uncertain boundary. isoPhon selects a random BAP band and then one of
the CU 10, 20, 30, or 40 boundary estimates from the same current posterior.

## 4. Audio Generation & Routing
Stimulus generation (Pure Tones, SAM-Tones, and Five-Tone Complexes) is calculated sample-by-sample in the C engine to ensure phase accuracy and precise RMS scaling.
When JavaScript requests a stimulus, the C engine writes the raw float data into shared memory. JavaScript reads this data, scales it to the target dB SPL, and feeds it into an `AudioBufferSourceNode`. The audio is then routed through a `StereoPannerNode` (directed by the UI's "Test Ear" dropdown) before reaching the hardware destination.

## 5. Tracking and Final Results
The application has two distinct stages. During data collection, a live
parametric Bayesian tracker maintains 21 candidate anchor-frequency models.
After each response, all 21 models are updated incrementally. The tracker then
selects the next stimulus from the current posterior: Bayesian mode targets
the most uncertain category boundary, while isoPhon selects a random BAP band
and one of the CU 10, 20, 30, or 40 boundary estimates. The tracker is used
for stimulus selection during the procedure, not for the final displayed fit.

After data collection is complete, the browser sends the full trial history to
the post-hoc PCA/MCPF maximum-likelihood fitter (`qcls_pca_fit_report`). This
fit estimates a 5-component PCA profile, and the resulting category boundaries
are displayed as **equal-loudness contours**: one curve per category boundary,
giving the level that elicits CU 5, 10, ... 50 as a function of frequency. The
current plot does not display uncertainty bands.

If an audiogram was entered it is drawn over the contours as a dashed line,
converted to dB SPL using the selected transducer's RETSPL. CU 5 and the
audiogram are meant to be the same construct, the lowest level the listener
reports hearing, so the gap between the two curves is directly readable.

> This replaces the earlier categorical heatmap, which painted a colour at
> every point of the frequency-by-level plane. That read as a dense measurement
> even where nothing was measured, and it had nowhere to put uncertainty.

The current interface displays one metric from the fitted contours:

- **Loudness Growth** (CU/dB): 45 CU between the CU 5 and CU 50 contours divided
  by the level range they span, averaged over bands.

The tracker and post-hoc fit are described in detail in:

- Shen, Y., Petersen, E. A., & Neely, S. T. (2024). Toward parametric Bayesian
  adaptive procedures for multi-frequency categorical loudness scaling. *The
  Journal of the Acoustical Society of America, 156*(1), 262-277.
- Shen, Y., Petersen, E. A., & Neely, S. T. (2026). Rapid profiling of loudness
  perception among older adults. *Ear and Hearing, 47*(3), 716.

## 6. Safety: Output Ceiling and Discomfort Backoff
**Max Output (dB SPL)** sets the loudest level that will ever be presented.
Default 100 dB SPL, and whatever is entered it is clamped to the 110 dB
equipment limit: 130 becomes 110, 20 becomes 50, blank becomes 100. It is
locked once a run starts.

The working ceiling also **falls automatically by 5 dB every time the listener
reports "Too Loud"**, in both phases, and never rises during a run. A final
guard sits immediately before level becomes sound, so no path can reach the
listener without passing a limit.

This matters because stimulus selection targets the least certain boundary, and
CU 50 sits near the top of the range: an unbounded run tends toward the loudest
levels rather than away from them. Lower the ceiling for a listener with
reduced tolerance.

## 7. Configuring the Audiogram Prior
Optional. Left blank, the procedure behaves exactly as it did before.

- **Nine threshold fields**, 250 Hz to 6 kHz. Leave any frequency blank if it
  was not measured; blanks fall back to that band's population distribution
  rather than to a guess. An entirely empty form uses the stock prior.
- **Units**: dB HL or dB SPL. **This is not a formality.** The prior was fitted
  against thresholds in dB SPL, so dB HL must have the RETSPL added.
- **Transducer**: selects which RETSPL table performs that conversion. On the
  same audiogram the CU 5 prior at 250 Hz lands at 67.9 dB SPL under the legacy
  table and 58.2 under ER-3A inserts, **9.7 dB apart**, at the frequency where
  the prior is most sensitive to threshold. Choosing wrongly biases the
  low-frequency prior silently.

> Only tables measured at BTNRH are offered. Nominal published figures for
> TDH-39, TDH-49/50 and HDA200 were removed after the ER-3A row taken from the
> same source proved 3.5 to 15.5 dB low against BTNRH's own measurements. To
> add a transducer, measure it rather than copying a standard.

Conditioning on the audiogram is worth 0.396 dB, SE 0.116, t = 3.4, on 148
listeners replaying recorded responses.

## 8. How to Use the App
1. **Load the App:** Open `index.html` in any modern web browser. The app needs
   a local server, such as Python's `http.server`, so WebAssembly can load.
2. **Configure:** Participant ID, Test Ear, Stimulus Type/Bandwidth, Start
   Level, and **Max Output**.
3. **Optional:** Enter the audiogram, and set units and transducer to match how
   it was measured.
4. **Calibrate:** Click **Calibrate** for a continuous 1 kHz reference tone
  labelled 60 dB SPL. Click again to stop. This is an operator check of the
  audio path; it does not alter numerical stimulus levels, contour estimates,
  or the audiogram prior.
5. **Run Test:** Click **Run Test** to begin Phase 1.
6. **Export:** On completion, two downloads are offered.
   - **Download trials (TBT)** is the raw input: phase, trial, frequency,
     level, response.
   - **Download contours (ELC)** is the result: 90 rows of band, frequency, CU,
     and fitted level. It does not currently include uncertainty estimates.
   Filenames are `qCLS_TBT_[ID]_[Ear]_[YYYYMMDD].csv` and `qCLS_ELC_...`.

## 9. Session Storage
Completed sessions are kept in the browser's `localStorage`, most recent first,
capped at 50. Selecting one from **Previous Sessions** restores its contours,
audiogram, trials and displayed growth metric. The saved mode label currently
distinguishes Bayesian from non-Bayesian sessions; isoPhon sessions are labeled
as Random when restored.

This is per browser and per machine, not a record system. It protects against a
lost reload, not a lost laptop. The CSV downloads remain the way results leave
the application.

## 10. Headless Simulation and Evaluation
The `sim/qcls_wasm_accuracy_sim.py` harness evaluates the compiled qCLS
WebAssembly implementation against simulated virtual listeners. It uses the
same C tracker and post-hoc PCA fitter as the webapp, while
`sim/wasm_runner.py` mirrors the session and adaptive-stimulus logic and
`sim/wasm_bridge.js` calls the Emscripten exports under Node.js.

Virtual-listener parameters and ground-truth CU 5 through CU 50 profiles are
stored in `sim/mcpf_parameters.csv`. `sim/mcpf_simulator.py` uses the MCPF
coefficients and false-alarm rates to draw categorical responses at each
presented frequency and level. The Python runner converts categories
0 through 10 to the webapp's response scale, 0 through 50 CU, before updating
the C tracker. The CSV profiles are the reference truth used for accuracy
metrics; response probabilities come from the MCPF parameters.

Requirements are Python with NumPy and pandas, Node.js, and the built
`qcls_core.js` / `qcls_core.wasm` files in the repository root. From the
repository root, run one mode with:

```bash
python sim/qcls_wasm_accuracy_sim.py --impl wasm --mode bayesian \
  --n_trials 60 --n_reps 20 --out_dir ./wasm_accuracy_results
```

Use `--mode random`, `--mode bayesian`, or `--mode isophon` to select the
webapp's Phase 2 mode. `--ground_truth_csv` can select another compatible
parameter/profile CSV. `--sweep_trials 20 40 60` runs cumulative sessions and
reports accuracy at those trial counts instead of only at the final count.
`--plot_repetitions` optionally displays fitted profiles as the simulation
runs. `--impl dummy` is only a harness/plumbing check; it is not an evaluation
of the qCLS webapp procedure.

The harness writes per-repetition MAE accuracy, pairwise repeat-session MAD
reliability, signed bias by listener/boundary/frequency, and a summary JSON
file. A convergence sweep writes its own table of mean, standard deviation,
and median MAE by trial count. These describe the procedure's numerical
contour recovery and repeatability for the supplied virtual-listener model;
they are not results from human participants.

### Evaluation Scope
The headless evaluation covers the numerical path from simulated categorical
responses through the live C tracker and post-hoc PCA fit, including the
webapp's default Phase 1 range-bounding rules, output-ceiling/backoff rules,
and all three Phase 2 selection modes. It does not exercise or assess:

- Web Audio stimulus generation, waveform bandwidth/type, SPL calibration,
  panning, hardware output, transducer response, or real listener perception.
- The browser UI, buttons, progress display, contour rendering, audiogram
  display, downloads, local session storage, or browser/device compatibility.
- Audiogram-conditioned priors. The headless runner uses the stock C prior;
  the browser's optional threshold entry, dB HL/SPL conversion, and transducer
  selection are not currently passed through the simulation protocol.
- Alternative UI settings for starting frequency, starting level, or maximum
  output. The headless runner uses the webapp defaults: 1 kHz, 50 dB SPL,
  100 dB SPL ceiling, and the 110 dB equipment maximum.
- Exact seeded equivalence with browser randomness. Random-mode selection uses
  Python's seeded RNG, while Bayesian/isoPhon selection uses C `rand()`; the
  session seed is not currently forwarded to the C RNG. None of these streams
  matches the browser's JavaScript `Math.random()` sequence.
