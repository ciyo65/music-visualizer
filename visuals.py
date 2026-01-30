import os

# Canonical list of visual styles used across the project
STYLES = [
    "Pulse Circle",
    "Waveform Bars",
    "Spectrum Helix",
    "Galaxy Particles",
    "Minimal Flash",
]

# Descriptions for each visual style (used in UI tooltips)
STYLE_DESCRIPTIONS = {
    "Pulse Circle": "A glowing circle that pulses with the beat. Great for electronic and bass-heavy music.",
    "Waveform Bars": "Classic equalizer bars that react to frequency bands. Works well with any genre.",
    "Spectrum Helix": "Rotating spiral visualization of the frequency spectrum. Best for ambient and progressive tracks.",
    "Galaxy Particles": "Starfield with center glow that zooms with bass hits. Perfect for cinematic and space-themed audio.",
    "Minimal Flash": "Clean black/white flashes on beat drops. Ideal for minimalist aesthetics and EDM.",
}

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
