import numpy as np
from moviepy.editor import VideoClip
import os

# Import canonical style list and helper
from visuals import STYLES, asset_filename

# Mock Audio Data for Preview
# We simulate a "beat"

def get_mock_data(t):
    # Simulate a kick drum beat every 0.5s
    vol = np.exp(-10 * ((t % 0.5)**2)) 
    
    # Simulate freq spectrum (high energy at low freq)
    freq_col = np.zeros(200)
    freq_col[:10] = vol * 2 # Bass
    freq_col[10:50] = vol * 0.5 # Mids
    freq_col[50:] = np.random.rand(150) * 0.1 * vol # Highs
    
    return vol, freq_col

# Re-implement draw_frame locally to avoid importing app (which has streamlit deps)
# We copy the exact logic from app.py
import math

def draw_frame(t, style, W, H):
    vol, freq_col = get_mock_data(t)
    
    # Initialize Canvas
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    if style == "Pulse Circle":
        bg = int(10 + (vol * 20))
        frame[:] = (bg, bg, bg+5)
        center = (W // 2, H // 2)
        max_radius = H // 2.5 
        radius = int((H // 6) + (vol * max_radius))
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist <= radius
        frame[mask] = [0, int(100 + (vol * 155)), int(200 + (vol * 55))]

    elif style == "Waveform Bars":
        frame[:] = (0, 0, 0)
        num_bars = 20 # Fewer bars for small preview
        bar_width = W // num_bars
        gap = 1
        chunk_size = len(freq_col) // num_bars
        
        for i in range(num_bars):
            start = i * chunk_size
            end = start + chunk_size
            mag = np.mean(freq_col[start:end]) if end > start else 0
            bar_h = int(mag * H * 0.9)
            x1 = i * bar_width + gap
            x2 = x1 + bar_width - gap
            y1 = H - bar_h
            if y1 < 0: y1 = 0
            c_r = int(128 * (1 - i/num_bars))
            c_g = int(255 * (i/num_bars))
            c_b = 255
            frame[y1:H, x1:x2] = [c_r, c_g, c_b]

    elif style == "Spectrum Helix":
        frame[:] = (10, 10, 30)
        center_x, center_y = W // 2, H // 2
        num_lines = 30
        max_radius = min(W, H) // 2 - 10
        base_radius = 20
        angle_offset = t * 2.0 
        
        Y, X = np.ogrid[:H, :W]
        Y, X = Y - center_y, X - center_x
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        Theta = (Theta - angle_offset) % (2 * np.pi)
        
        freq_indices = (Theta / (2 * np.pi) * len(freq_col)).astype(int)
        freq_indices = np.clip(freq_indices, 0, len(freq_col)-1)
        mags = freq_col[freq_indices]
        
        outer_limit = base_radius + mags * (max_radius - base_radius) * 1.5
        mask = (R > base_radius) & (R < outer_limit)
        hue = (Theta / (2 * np.pi)) 
        frame[mask, 0] = (np.sin(hue[mask] * 6.28) * 127 + 128).astype(np.uint8)
        frame[mask, 1] = (np.sin(hue[mask] * 6.28 + 2) * 127 + 128).astype(np.uint8) 
        frame[mask, 2] = 255
        
    elif style == "Galaxy Particles":
        frame[:] = (5, 5, 10)
        center_x, center_y = W // 2, H // 2
        np.random.seed(42) 
        num_stars = 100
        star_x = np.random.randint(0, W, num_stars)
        star_y = np.random.randint(0, H, num_stars)
        bass = vol 
        zoom = 1.0 + (bass * 0.5)
        shifted_x = (star_x - center_x) * zoom + center_x
        shifted_y = (star_y - center_y) * zoom + center_y
        valid = (shifted_x >= 0) & (shifted_x < W) & (shifted_y >= 0) & (shifted_y < H)
        sx = shifted_x[valid].astype(int)
        sy = shifted_y[valid].astype(int)
        frame[sy, sx] = [255, 255, 200]
        
        core_radius = int(20 * (1 + vol))
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        glow_mask = dist < (core_radius * 2)
        if np.any(glow_mask):
            alpha = 1 - (dist[glow_mask] / (core_radius * 2))
            alpha = np.clip(alpha, 0, 1)
            frame[glow_mask, 0] = np.clip(frame[glow_mask, 0] + alpha * 255 * vol, 0, 255) 
            frame[glow_mask, 1] = np.clip(frame[glow_mask, 1] + alpha * 100 * vol, 0, 255) 
            frame[glow_mask, 2] = np.clip(frame[glow_mask, 2] + alpha * 200 * vol, 0, 255) 

    elif style == "Minimal Flash":
        c = 255 if vol > 0.65 else int(vol * 40)
        frame[:] = (c, c, c)

    return frame


def generate_previews(out_dir="assets", duration=2.0, fps=15, sr=22050, W=300, H=200):
    """Render preview GIFs for all styles into out_dir.

    This is intentionally fast and uses small W/H and low fps by default.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Generating previews...")

    for style in STYLES:
        filename = asset_filename(style, out_dir=out_dir)
        print(f"Rendering {filename}...")

        def make_frame(t):
            return draw_frame(t, style, W, H)

        clip = VideoClip(make_frame, duration=duration)
        clip.write_gif(filename, fps=fps, logger=None)

    print("Done!")


if __name__ == "__main__":
    generate_previews()
