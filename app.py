import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import uuid
import math
from moviepy.editor import VideoClip, AudioFileClip

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Music Visualizer", page_icon="🎵")

st.title("🎵 AI Music Visualizer")
st.write("Upload a track (MP3, WAV, AAC, etc.) or video (MP4, MOV), pick a style, and generate your video.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Visual Style
    visual_style = st.selectbox(
        "Choose Visual Style",
        ["Pulse Circle", "Waveform Bars", "Spectrum Helix", "Galaxy Particles", "Minimal Flash"]
    )
    
    # 2. Resolution (Crucial for Cloud Stability)
    resolution_mode = st.selectbox(
        "Video Resolution",
        ["Mobile Low (480p)", "HD (720p)", "Full HD (1080p)"],
        index=0 # Default to 480p to prevent crashes
    )
    
    # 3. Duration Limit
    duration_mode = st.radio("Render Length", ["Preview (30s)", "Full Song"])

    st.info("💡 **Tip:** Use 'Mobile Low' or 'HD' for faster rendering on the free cloud server.")

# --- AUDIO PROCESSING ---
def get_audio_features(file_path):
    """Loads audio and returns volume (RMS) and frequency spectrogram."""
    # librosa.load supports wav, mp3, ogg, au, flac, etc. (via ffmpeg)
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    hop_length = 512
    # 1. Volume (RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    # Handle silence/zeros to avoid divide by zero
    if np.max(rms) > np.min(rms):
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    else:
        rms_norm = rms

    # 2. Frequency (Spectrogram)
    # n_fft=2048 is standard for music analysis
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    # Convert to dB for better visual scaling
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    
    # Normalize -80dB to 0dB range into 0.0 to 1.0
    D_db_norm = (D_db + 80) / 80
    D_db_norm = np.clip(D_db_norm, 0, 1)
    
    return y, sr, duration, rms_norm, D_db_norm

# --- DRAWING ENGINE ---
def draw_frame(t, style, rms_norm, spec_norm, sr, W, H):
    hop_length = 512
    frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
    
    # Safety check for index out of bounds
    if frame_idx >= len(rms_norm):
        vol = 0
    else:
        vol = rms_norm[frame_idx]

    # Initialize Canvas
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Common spectrogram data retrieval
    if frame_idx < spec_norm.shape[1]:
        freq_col = spec_norm[:200, frame_idx] # First 200 bins
    else:
        freq_col = np.zeros(200)

    if style == "Pulse Circle":
        # Dynamic Background
        bg = int(10 + (vol * 20))
        frame[:] = (bg, bg, bg+5)
        
        # Circle Logic
        center = (W // 2, H // 2)
        # Scale radius based on resolution height
        max_radius = H // 2.5 
        radius = int((H // 6) + (vol * max_radius))
        
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist <= radius
        
        frame[mask] = [0, int(100 + (vol * 155)), int(200 + (vol * 55))]

    elif style == "Waveform Bars":
        # Black Background
        frame[:] = (0, 0, 0)
        
        num_bars = 40
        bar_width = W // num_bars
        gap = 2
        
        chunk_size = len(freq_col) // num_bars
        
        for i in range(num_bars):
            # Get average magnitude for this frequency band
            start = i * chunk_size
            end = start + chunk_size
            mag = np.mean(freq_col[start:end]) if end > start else 0
            
            # Boost height for visual impact
            bar_h = int(mag * H * 0.9)
            
            x1 = i * bar_width + gap
            x2 = x1 + bar_width - gap
            y1 = H - bar_h
            if y1 < 0: y1 = 0
            
            # Use a colorful gradient: Purple -> Blue -> Cyan
            c_r = int(128 * (1 - i/num_bars))
            c_g = int(255 * (i/num_bars))
            c_b = 255
            
            frame[y1:H, x1:x2] = [c_r, c_g, c_b]

    elif style == "Spectrum Helix":
        # Dark blue background
        frame[:] = (10, 10, 30)
        
        center_x, center_y = W // 2, H // 2
        num_lines = 60
        max_radius = min(W, H) // 2 - 20
        base_radius = 50
        
        # Resample freq_col to num_lines
        chunk_size = len(freq_col) // num_lines
        if chunk_size < 1: chunk_size = 1
        
        # Rotate the whole helix over time
        angle_offset = t * 0.5 
        
        for i in range(num_lines):
            # Get magnitude
            idx = i * chunk_size
            if idx < len(freq_col):
                mag = freq_col[idx]
            else:
                mag = 0
            
            # Calculate angle
            angle = angle_offset + (i / num_lines) * 2 * np.pi
            
            # Line length based on magnitude
            line_len = int(mag * (max_radius - base_radius))
            
            # Start and End points
            x_start = int(center_x + math.cos(angle) * base_radius)
            y_start = int(center_y + math.sin(angle) * base_radius)
            
            x_end = int(center_x + math.cos(angle) * (base_radius + line_len))
            y_end = int(center_y + math.sin(angle) * (base_radius + line_len))
            
            # Draw approx line using Bresenham algorithm or simplified iteration (for pure numpy)
            # Since numpy doesn't have a draw_line, we can simulate thick points or use RR 
            # But RR (skimage.draw.line) isn't imported. 
            # Let's use a simpler approach: multiple dots or small rectangles along the path.
            # OR better: a simple "fan" of pixels if we want pure numpy speed, 
            # but for clarity let's just draw a small block at the end position for a "particle ring" effect
            # to keep it fast without CV2/PIL.
            
            # Let's draw "rays" using a mask for a wedge? No, too slow.
            # Let's simple fill a small rectangular area at the calculated position?
            
            # Let's implement a simple line drawer:
            # Actually, standard python loops are slow for pixels. 
            # A vectorized approach for "Helix" is better:
            # Create a radial grid.
            
            pass # We will use a vectorized approach below loop
            
        # Vectorized Helix
        Y, X = np.ogrid[:H, :W]
        # Coordinates relative to center
        Y, X = Y - center_y, X - center_x
        # Convert to polar
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Normalize Theta to 0..2PI
        Theta = (Theta - angle_offset) % (2 * np.pi)
        
        # Map angle to frequency index
        # 0..2PI -> 0..len(freq_col)
        freq_indices = (Theta / (2 * np.pi) * len(freq_col)).astype(int)
        freq_indices = np.clip(freq_indices, 0, len(freq_col)-1)
        
        # Get magnitudes for all pixels based on angle
        mags = freq_col[freq_indices]
        
        # Define ring boundaries
        # Pixels are "on" if R is between base_radius and (base + mag * max_scale)
        outer_limit = base_radius + mags * (max_radius - base_radius) * 1.5
        
        mask = (R > base_radius) & (R < outer_limit)
        
        # Coloring based on angle
        # Simple colorful map
        hue = (Theta / (2 * np.pi)) 
        # R = sin(hue), G = sin(hue + 1/3), B = sin(hue + 2/3) approx
        
        frame[mask, 0] = (np.sin(hue[mask] * 6.28) * 127 + 128).astype(np.uint8)
        frame[mask, 1] = (np.sin(hue[mask] * 6.28 + 2) * 127 + 128).astype(np.uint8) 
        frame[mask, 2] = 255
        
    elif style == "Galaxy Particles":
        # Starfield background
        frame[:] = (5, 5, 10)
        
        # Use a pseudo-random number generator seeded with time for consistency across frames?
        # No, for particles we want continuity. But since we don't have state persistence across frames in this simple func,
        # we can simulate "random" positions based on hashing coordinates or simple noise.
        # However, for a "Galaxy" effect that pulses with music, we can use a static set of stars 
        # that scale out from center based on bass.
        
        center_x, center_y = W // 2, H // 2
        
        # Generate static star positions based on a fixed seed
        np.random.seed(42) 
        num_stars = 200
        star_x = np.random.randint(0, W, num_stars)
        star_y = np.random.randint(0, H, num_stars)
        star_sizes = np.random.randint(1, 4, num_stars)
        
        # Bass kick factor (low frequencies)
        bass = np.mean(freq_col[:10]) 
        zoom = 1.0 + (bass * 0.5)
        
        # Shift stars away from center based on zoom
        # (X - cx) * zoom + cx
        shifted_x = (star_x - center_x) * zoom + center_x
        shifted_y = (star_y - center_y) * zoom + center_y
        
        # Filter out of bounds
        valid = (shifted_x >= 0) & (shifted_x < W) & (shifted_y >= 0) & (shifted_y < H)
        
        sx = shifted_x[valid].astype(int)
        sy = shifted_y[valid].astype(int)
        
        # Draw stars
        # Since we can't easily loop, let's just set pixels. 
        # For larger stars, we'd need a loop or dilation, but single pixels are fine for "distant stars"
        frame[sy, sx] = [255, 255, 200]
        
        # Add a central "Core" that glows with volume
        core_radius = int(50 * (1 + vol))
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Soft glow
        glow_mask = dist < (core_radius * 2)
        if np.any(glow_mask):
            # Inverse distance weighting for alpha
            alpha = 1 - (dist[glow_mask] / (core_radius * 2))
            alpha = np.clip(alpha, 0, 1)
            
            # Additive blending
            frame[glow_mask, 0] = np.clip(frame[glow_mask, 0] + alpha * 255 * vol, 0, 255) # Red
            frame[glow_mask, 1] = np.clip(frame[glow_mask, 1] + alpha * 100 * vol, 0, 255) # Orange-ish
            frame[glow_mask, 2] = np.clip(frame[glow_mask, 2] + alpha * 200 * vol, 0, 255) # Purple tint

    elif style == "Minimal Flash":
        c = 255 if vol > 0.65 else int(vol * 40)
        frame[:] = (c, c, c)

    return frame

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader(
    "Choose a file (Audio or Video)", 
    type=["mp3", "wav", "ogg", "flac", "aac", "m4a", "mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    # Display audio player regardless of format (streamlit converts internally or browser handles it)
    # For video files, st.audio works if it has an audio track, but st.video is better for previewing input video.
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type in ['mp4', 'mov', 'avi', 'mkv']:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file, format=f'audio/{file_type}')
    
    if st.button("🎬 Generate Video"):
        with st.status("Processing...", expanded=True) as status:
            
            # 1. Save temp input file
            # We need to preserve extension for tools to recognize format
            ext = "." + file_type
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext) 
            tfile.write(uploaded_file.read())
            temp_input_path = tfile.name
            tfile.close() 

            output_video_path = f"output_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                # 2. Analyze Audio
                status.write("🎵 Analyzing Audio Data (FFT)...")
                
                # librosa.load will extract audio from video files automatically via ffmpeg
                y, sr, total_duration, rms_norm, spec_norm = get_audio_features(temp_input_path)
                
                # Duration Logic
                render_duration = 30 if duration_mode == "Preview (30s)" else total_duration
                if render_duration > total_duration: render_duration = total_duration

                # 3. Set Resolution
                if resolution_mode == "Mobile Low (480p)":
                    W, H = 854, 480
                elif resolution_mode == "HD (720p)":
                    W, H = 1280, 720
                else:
                    W, H = 1920, 1080

                # 4. Render
                status.write(f"🎨 Rendering at {W}x{H}...")
                
                def make_frame_wrapper(t):
                    return draw_frame(t, visual_style, rms_norm, spec_norm, sr, W, H)
                
                clip = VideoClip(make_frame_wrapper, duration=render_duration)
                
                # Extract audio from the input file (works for both audio and video inputs)
                audio_clip = AudioFileClip(temp_input_path).subclip(0, render_duration)
                clip = clip.set_audio(audio_clip)
                
                clip.write_videofile(
                    output_video_path, 
                    fps=24, 
                    codec='libx264', 
                    audio_codec='aac', 
                    preset='ultrafast',
                    logger=None 
                )
                
                status.update(label="✅ Complete!", state="complete", expanded=False)
                
                # 5. Show Results
                st.success(f"Video Generated! ({W}x{H})")
                st.video(output_video_path)
                
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=file,
                        file_name="visualizer_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                # Cleanup
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
