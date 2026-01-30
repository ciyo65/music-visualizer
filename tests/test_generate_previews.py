import shutil
import os
import pytest

from pathlib import Path

def test_generate_previews_creates_assets(tmp_path):
    # Skip on systems without ffmpeg to avoid noisy failures; CI installs ffmpeg explicitly
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    out_dir = tmp_path / "assets"
    from generate_previews import generate_previews
    generate_previews(out_dir=str(out_dir), duration=0.5, fps=5, W=64, H=32)

    from visuals import STYLES
    for style in STYLES:
        expected = out_dir / f"{style.lower().replace(' ', '_')}.gif"
        assert expected.exists(), f"Missing preview asset: {expected}"
