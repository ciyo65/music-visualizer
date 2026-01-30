import streamlit as st
import librosa
import numpy as np
import tempfile
import os
from moviepy.editor import VideoClip, AudioFileClip

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Music Visualizer", page_icon="🎵")

st.title("🎵 AI Music Visualizer")
st.write("Upload a track, pick a style, and generate your video.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Visual Style
    visual_style = st.selectbox(
        "Choose Visual Style",
        ["Pulse Circle", "Waveform Bars", "Minimal Flash"]
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
    """Loads audio and returns the normalized volume (RMS) array."""
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    
    return y, sr, duration, rms_norm

# --- DRAWING ENGINE ---
def draw_frame(t, style, rms_norm, sr, W, H):
    # Get volume at current time 't'
    hop_length = 512
    frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
    vol = rms_norm[frame_idx] if frame_idx < len(rms_norm) else 0

    # Initialize Canvas
    frame = np.zeros((H, W, 3), dtype=np.uint8)

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
        
        num_bars = 30
        bar_width = W // num_bars
        np.random.seed(int(t * 10)) 
        
        for i in range(num_bars):
            h_factor = np.random.rand()
            # Scale height based on resolution height
            bar_h = int((vol * (H * 0.9)) * h_factor) + 10
            
            x1 = i * bar_width + 2
            x2 = x1 + bar_width - 4
            y1 = H - bar_h
            
            frame[y1:H, x1:x2] = [0, 255, 100]

    elif style == "Minimal Flash":
        c = 255 if vol > 0.65 else int(vol * 40)
        frame[:] = (c, c, c)

    return frame

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    if st.button("🎬 Generate Video"):
        with st.status("Processing...", expanded=True) as status:
            
            # 1. Save temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") 
            tfile.write(uploaded_file.read())
            temp_input_path = tfile.name
            
            # 2. Analyze Audio
            status.write("🎵 Analyzing Audio Data...")
            y, sr, total_duration, rms_norm = get_audio_features(temp_input_path)
            
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
            
            output_video_path = "output_video.mp4"
            
            def make_frame_wrapper(t):
                return draw_frame(t, visual_style, rms_norm, sr, W, H)
            
            clip = VideoClip(make_frame_wrapper, duration=render_duration)
            audio_clip = AudioFileClip(temp_input_path).subclip(0, render_duration)
            clip = clip.set_audio(audio_clip)
            
            clip.write_videofile(
                output_video_path, 
                fps=24, 
                codec='libx264', 
                audio_codec='aac', 
                preset='ultrafast', # Keeps render fast
                logger=None 
            )
            
            os.remove(temp_input_path)
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