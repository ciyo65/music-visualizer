# Copilot / Agent Instructions for AI contributors ✅

Purpose
- Short, actionable guidance for AI coding agents to be productive here.

Quick Start
1. Install system deps: ffmpeg is required (moviepy uses ffmpeg). On macOS: `brew install ffmpeg` ✅
2. Install Python deps: `pip install -r requirements.txt` or see `packages.txt` for extra packages.
3. Run the app locally: `streamlit run app.py` → open the URL shown in the terminal.
4. Re-generate preview assets: `python generate_previews.py` (writes GIFs to `assets/`).

Big picture (what this project is)
- Single-page Streamlit app that turns uploaded audio into short visualizer videos (`app.py`).
- A small helper script (`generate_previews.py`) creates lightweight GIF previews used by the UI.
- Static assets live in `assets/` (preview GIFs named after styles like `pulse_circle.gif`).

Key files
- `app.py` → main Streamlit app, UI + audio processing + video rendering.
- `generate_previews.py` → isolated preview renderer used for quick checks and to build `assets/` GIFs.
- `requirements.txt` / `packages.txt` → Python package lists.

Important conventions & patterns (must preserve)
- Visual styles are now centralized in `visuals.py` as `STYLES` and `asset_filename(...)` — use this single source of truth for labels and asset paths.
  - Example: `asset_filename('Pulse Circle')` -> `assets/pulse_circle.gif`.
- `draw_frame(...)` is still duplicated between `app.py` and `generate_previews.py` (intentional to avoid Streamlit import in the preview generator). When changing visuals, update both places or extract common rendering helpers carefully.
- Audio processing choices are explicit and should be preserved:
  - `librosa` for loading + RMS (`hop_length=512`) and STFT (`n_fft=2048`).
  - Preview generation uses small defaults for fast iteration (`fps` low, small `W/H`) while the final exporter uses `fps = 24` and the user-selected resolution presets.

Runtime notes & gotchas ⚠️
- moviepy requires a working `ffmpeg` binary; missing ffmpeg causes silent failures during export.
- Large audio files can consume a lot of memory/time. Use the sidebar "Preview (30s)" + "Mobile Low (480p)" during iteration.
- The app writes an output file `output_<uuid>.mp4` into the working dir and cleans temp input files; watch for leftover files when debugging.

Recommended quick workflows for agents
- To iterate on visual frames: modify `draw_frame` in `generate_previews.py`, run `python generate_previews.py` (fast defaults), and inspect `assets/*.gif`. When happy, port changes to `app.py` or keep both in sync.
- To run the automated check: `pytest -q` will run `tests/test_generate_previews.py` (fast, skips if `ffmpeg` is missing). CI (`.github/workflows/ci.yml`) installs `ffmpeg` and runs the tests on push/PR.

Good-first PR opportunities for agents
- Add more unit tests for `draw_frame` outputs (assert shapes, dtypes, and simple pixel checks).
- Add a smoke test that starts the Streamlit app and requests the `/` endpoint (simple health check).
- Improve error messaging when `ffmpeg` is not available (there is already a CI job that installs it; local dev should show a clear note).

What NOT to change without more context
- UI text and preset labels (used directly by the UI and previews) — changes must be mirrored in both places.
- Audio analysis parameters (hop_length / n_fft) without testing perceived visual timing and artifacting.

If anything here is unclear or missing, please point it out so I can iterate. 🛠️
