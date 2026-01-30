import os

# Canonical list of visual styles used across the project
STYLES = [
    "Pulse Circle",
    "Waveform Bars",
    "Spectrum Helix",
    "Galaxy Particles",
    "Minimal Flash",
]

def asset_filename(style: str, out_dir: str = "assets") -> str:
    """Return the filesystem path for a style's preview asset.

    Example: asset_filename('Pulse Circle') -> 'assets/pulse_circle.gif'
    """
    fname = f"{style.lower().replace(' ', '_')}.gif"
    return os.path.join(out_dir, fname)
