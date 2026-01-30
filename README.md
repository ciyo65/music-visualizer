# Music Visualizer

AI Music Visualizer is a single-page Streamlit app that converts uploaded audio into short visualizer videos.

Highlights
- Streamlit UI (`app.py`) for uploading audio and generating MP4 visualizers.
- Fast preview generator (`generate_previews.py`) that renders small GIFs into `assets/` for the UI.
- Shared visual style constants centralized in `visuals.py` and used by both scripts.

Quick start
1. Install system deps (macOS):
   - ffmpeg is required by `moviepy`: `brew install ffmpeg` ✅
2. Create a Python environment and install dependencies:
   - `pip install -r requirements.txt`
3. Run the app locally:
   - `streamlit run app.py`
4. Regenerate preview GIFs (fast):
   - `python generate_previews.py` or `python -m generate_previews`

Testing
- Unit tests live in `tests/`.
- Quick previews test: `pytest tests/test_generate_previews.py` (skips if `ffmpeg` missing).
- UI smoke test: `tests/test_streamlit_smoke.py` (requires `streamlit` CLI and `requests`).

Developer notes & conventions
- Visual styles are the canonical `STYLES` located in `visuals.py`. Use `asset_filename(style)` to get the preview asset path (e.g., `assets/pulse_circle.gif`).
- `draw_frame(...)` is duplicated in `app.py` and `generate_previews.py` (intentional to avoid importing Streamlit). If you change visual logic, update both locations or extract shared helpers carefully.
- Audio analysis parameters are explicit and should be preserved where possible (`librosa` with `hop_length=512`, `n_fft=2048`).
- The app writes outputs as `output_<uuid>.mp4` in the working dir and removes temporary input files after rendering.

CI
- The repository includes `.github/workflows/ci.yml` which installs `ffmpeg` on CI, installs dependencies, and runs tests on push/PR.

Useful scripts
- `scripts/run_smoke.sh`: run Streamlit headless for manual smoke testing.

Contributing
- Open a PR against `main`. Small, focused PRs are preferred (one change per PR).
- Update `assets/` GIFs when changing visuals, and ensure `visuals.py` stays authoritative for labels.

If anything in this README is inaccurate or missing, please open an issue or a PR with suggested changes.