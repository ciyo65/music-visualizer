import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import uuid
from moviepy.editor import VideoClip, AudioFileClip

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Music Visualizer",
    page_icon=":material/equalizer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM UI CSS ---
st.markdown("""
<style>
    /* Global Reset & Font */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1d1d1f;
    }
    
    /* Remove default streamlit junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background */
    .stApp {
        background-color: #fbfbfd; /* Apple-like off-white */
    }

    /* CARD COMPONENT */
    .css-card {
        background: white;
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.02);
        margin-bottom: 24px;
    }
    
    /* HEADERS */
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h3 {
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        color: #86868b;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: #0071e3; /* Apple Blue */
        color: white;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 980px; /* Pill shape */
        border: none;
        box-shadow: 0 4px 12px rgba(0,113,227,0.2);
        transition: transform 0.1s ease, box-shadow 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #0077ed;
        transform: scale(1.01);
        box-shadow: 0 6px 16px rgba(0,113,227,0.3);
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        padding: 2rem;
        border: 2px dashed #d2d2d7;
        border-radius: 12px;
        background-color: #f5f5f7;
        text-align: center;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #0071e3;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    
    /* VIDEO/AUDIO CONTAINERS */
    .stVideo, .stAudio {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>AI Music Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.1rem;'>Transform your audio into studio-quality reactive visuals.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.markdown("### :material/tune: Configuration")
    
    # Visual Style
    st.markdown("**Visual Theme**")
    style_options = ["Pulse Circle", "Waveform Bars", "Spectrum Helix", "Galaxy Particles", "Minimal Flash"]
    visual_style = st.selectbox("Style", style_options, label_visibility="collapsed")
    
    # Preview GIF in a clean container
    preview_path = f"assets/{visual_style.lower().replace(' ', '_')}.gif"
    if os.path.exists(preview_path):
        st.image(preview_path, use_container_width=True)
        st.caption(f"Preview: {visual_style}")

    st.markdown("---")
    
    # Resolution
    st.markdown("**Output Quality**")
    resolution_mode = st.selectbox(
        "Resolution",
        ["Mobile Low (480p)", "HD (720p)", "Full HD (1080p)"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Duration
    st.markdown("**Duration**")
    duration_mode = st.radio(
        "Duration", 
        ["Preview (30s)", "Full Song"],
        label_visibility="collapsed"
    )

# --- AUDIO PROCESSING FUNCTIONS ---
def get_audio_features(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if np.max(rms) > np.min(rms):
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    else:
        rms_norm = rms
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    D_db_norm = (D_db + 80) / 80
    D_db_norm = np.clip(D_db_norm, 0, 1)
    return y, sr, duration, rms_norm, D_db_norm

def draw_frame(t, style, rms_norm, spec_norm, sr, W, H):
    hop_length = 512
    frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
    vol = rms_norm[frame_idx] if frame_idx < len(rms_norm) else 0
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    if frame_idx < spec_norm.shape[1]:
        freq_col = spec_norm[:200, frame_idx]
    else:
        freq_col = np.zeros(200)

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
        num_bars = 40
        bar_width = W // num_bars
        gap = 2
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
        max_radius = min(W, H) // 2 - 20
        base_radius = 50
        angle_offset = t * 0.5 
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
        num_stars = 200
        star_x = np.random.randint(0, W, num_stars)
        star_y = np.random.randint(0, H, num_stars)
        bass = np.mean(freq_col[:10]) 
        zoom = 1.0 + (bass * 0.5)
        shifted_x = (star_x - center_x) * zoom + center_x
        shifted_y = (star_y - center_y) * zoom + center_y
        valid = (shifted_x >= 0) & (shifted_x < W) & (shifted_y >= 0) & (shifted_y < H)
        sx = shifted_x[valid].astype(int)
        sy = shifted_y[valid].astype(int)
        frame[sy, sx] = [255, 255, 200]
        core_radius = int(50 * (1 + vol))
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

# --- MAIN LAYOUT ---

# 1. UPLOAD SECTION
# Using 3 columns to center the upload box and prevent it from being too wide
u_col1, u_col2, u_col3 = st.columns([1, 2, 1])
with u_col2:
    uploaded_file = st.file_uploader(
        "Upload Audio", 
        type=["mp3", "wav", "ogg", "flac", "aac", "m4a", "mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed"
    )
    if not uploaded_file:
        st.info("👆 Drag & drop an audio file to get started")

if uploaded_file is not None:
    # 2. PREVIEW SECTION
    st.markdown("---")
    st.markdown("### :material/play_circle: Source Preview")
    
    # Grid layout to constrain preview size (Proportion fix)
    p_col1, p_col2, p_col3 = st.columns([1, 2, 1]) 
    with p_col2:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type in ['mp4', 'mov', 'avi', 'mkv']:
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file, format=f'audio/{file_type}')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. ACTION SECTION
    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    with b_col2:
        generate_btn = st.button("Generate Video", icon="✨", use_container_width=True)

    if generate_btn:
        # Create a container for the process to isolate it visually
        process_container = st.container()
        
        with process_container:
             with st.status("Processing your masterpiece...", expanded=True) as status:
                ext = "." + file_type
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext) 
                tfile.write(uploaded_file.read())
                temp_input_path = tfile.name
                tfile.close() 
                output_video_path = f"output_{uuid.uuid4().hex[:8]}.mp4"
                
                try:
                    status.write(":material/analytics: Analyzing audio spectrum...")
                    y, sr, total_duration, rms_norm, spec_norm = get_audio_features(temp_input_path)
                    
                    render_duration = 30 if duration_mode == "Preview (30s)" else total_duration
                    if render_duration > total_duration: render_duration = total_duration
                    
                    if resolution_mode == "Mobile Low (480p)": W, H = 854, 480
                    elif resolution_mode == "HD (720p)": W, H = 1280, 720
                    else: W, H = 1920, 1080
                    
                    status.write(f":material/brush: Rendering visuals ({W}x{H})...")
                    def make_frame_wrapper(t):
                        return draw_frame(t, visual_style, rms_norm, spec_norm, sr, W, H)
                    
                    clip = VideoClip(make_frame_wrapper, duration=render_duration)
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
                    
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.balloons()
                    
                    # 4. RESULT SECTION (Centered and constrained)
                    st.markdown("### :material/movie: Final Render")
                    r_col1, r_col2, r_col3 = st.columns([1, 2, 1])
                    with r_col2:
                        st.video(output_video_path)
                        
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="Download Video",
                                data=file,
                                file_name="visualizer_video.mp4",
                                mime="video/mp4",
                                icon=":material/download:",
                                use_container_width=True
                            )
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(temp_input_path):
                        os.remove(temp_input_path)
