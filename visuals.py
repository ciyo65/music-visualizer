import os

# Canonical list of visual styles used across the project
STYLES = [
    "Pulse Circle",
    "Waveform Bars",
    "Spectrum Helix",
    "Galaxy Particles",
    "Minimal Flash",
]

# Platform presets for orientation and aspect ratio
PLATFORMS = {
    "YouTube & TV (16:9)": "Landscape (16:9)",
    "TikTok, Reels & Shorts (9:16)": "Portrait (9:16)",
}

def asset_filename(style: str, out_dir: str = "assets") -> str:
    """Return the filesystem path for a style's preview asset.

    Example: asset_filename('Pulse Circle') -> 'assets/pulse_circle.gif'
    """
    fname = f"{style.lower().replace(' ', '_')}.gif"
    return os.path.join(out_dir, fname)