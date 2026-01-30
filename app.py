import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import uuid
from moviepy.editor import VideoClip, AudioFileClip

# Shared visual style constants
from visuals import STYLES, PLATFORMS, asset_filename

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
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #fbfbfd;
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
    
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    /* Subtitle Contrast Fix */
    .subtitle {
        text-align: center; 
        color: #555; 
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Input Focus States for Accessibility */
    .stSelectbox:focus-within, .stRadio:focus-within, .stFileUploader:focus-within {
        outline: 2px solid #0071e3;
        outline-offset: 4px;
        border-radius: 4px;
    }

    /* Standardized Button Padding */
    .stButton > button, .stDownloadButton > button {
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 980px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        border: none !important;
    }

    .stButton > button {
        background: #0071e3 !important;
        color: white !important;
    }
    
    .stDownloadButton > button {
        background: #34a853 !important;
        color: white !important;
    }

    .stButton > button:hover {
        background: #0077ed !important;
        box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important;
    }

    .stDownloadButton > button:hover {
        background: #2d8e47 !important;
        box-shadow: 0 4px 12px rgba(52,168,83,0.3) !important;
    }

    /* Mobile specific tweaks */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem; }
        .stButton > button { padding: 1rem; }
        /* Fix cramped columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("<h1 style='text-align: center;'>:material/equalizer: AI Music Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Turn audio into reactive visuals in seconds.</p>", unsafe_allow_html=True)

# --- ADVANCED SETTINGS (SIDEBAR) ---
with st.sidebar:
    st.markdown("### :material/settings: Advanced")
    resolution_mode = st.selectbox(
        "Resolution",
        ["Mobile Low (480p)", "HD (720p)", "Full HD (1080p)"],
        index=0,
        help="Higher resolutions take longer to render."
    )
    duration_mode = st.radio(
        "Duration", 
        ["Preview (30s)", "Full Song"],
        help="Preview is best for testing styles quickly."
    )
    st.info(":material/lightbulb: Pro Tip: Use 'Mobile Low' for faster rendering on phone.")

# --- AUDIO PROCESSING FUNCTIONS ---
def get_audio_features(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms)) if np.max(rms) > np.min(rms) else rms
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    D_db_norm = np.clip((D_db + 80) / 80, 0, 1)
    return y, sr, duration, rms_norm, D_db_norm

def draw_frame(t, style, rms_norm, spec_norm, sr, W, H):
    hop_length = 512
    frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
    vol = rms_norm[frame_idx] if frame_idx < len(rms_norm) else 0
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    freq_col = spec_norm[:200, frame_idx] if frame_idx < spec_norm.shape[1] else np.zeros(200)

    if style == "Pulse Circle":
        bg = int(10 + (vol * 20)); frame[:] = (bg, bg, bg+5)
        center = (W // 2, H // 2)
        dim = min(W, H)
        radius = int((dim // 6) + (vol * (dim // 2.5)))
        Y, X = np.ogrid[:H, :W]
        mask = np.sqrt((X - center[0])**2 + (Y - center[1])**2) <= radius
        frame[mask] = [0, int(100 + (vol * 155)), int(200 + (vol * 55))]
    elif style == "Waveform Bars":
        num_bars = 40; bar_width = W // num_bars; chunk_size = len(freq_col) // num_bars
        for i in range(num_bars):
            mag = np.mean(freq_col[i*chunk_size:(i+1)*chunk_size])
            bar_h = int(mag * H * 0.9)
            x1, x2, y1 = i*bar_width+2, (i+1)*bar_width-2, max(0, H-bar_h)
            frame[y1:H, x1:x2] = [int(128*(1-i/num_bars)), int(255*(i/num_bars)), 255]
    elif style == "Spectrum Helix":
        frame[:] = (10, 10, 30); center_x, center_y = W // 2, H // 2
        Y, X = np.ogrid[:H, :W]; Y, X = Y - center_y, X - center_x
        R, Theta = np.sqrt(X**2 + Y**2), (np.arctan2(Y, X) - t*0.5) % (2*np.pi)
        freq_idx = np.clip((Theta/(2*np.pi)*len(freq_col)).astype(int), 0, len(freq_col)-1)
        mask = (R > 50) & (R < 50 + freq_col[freq_idx]*(min(W,H)//2-20)*1.5)
        hue = Theta / (2 * np.pi)
        frame[mask, 0] = (np.sin(hue[mask]*6.28)*127+128).astype(np.uint8)
        frame[mask, 1] = (np.sin(hue[mask]*6.28+2)*127+128).astype(np.uint8)
        frame[mask, 2] = 255
    elif style == "Galaxy Particles":
        frame[:] = (5, 5, 10); np.random.seed(42)
        star_x, star_y = np.random.randint(0, W, 200), np.random.randint(0, H, 200)
        zoom = 1.0 + (np.mean(freq_col[:10]) * 0.5)
        sx, sy = ((star_x-W//2)*zoom+W//2).astype(int), ((star_y-H//2)*zoom+H//2).astype(int)
        valid = (sx>=0)&(sx<W)&(sy>=0)&(sy<H)
        frame[sy[valid], sx[valid]] = [255, 255, 200]
        Y, X = np.ogrid[:H, :W]; dist = np.sqrt((X-W//2)**2 + (Y-H//2)**2)
        glow_mask = dist < 50*(1+vol)*2
        if np.any(glow_mask):
            alpha = np.clip(1-(dist[glow_mask]/(50*(1+vol)*2)), 0, 1)
            frame[glow_mask, 0] = np.clip(frame[glow_mask,0]+alpha*255*vol, 0, 255)
            frame[glow_mask, 1] = np.clip(frame[glow_mask,1]+alpha*100*vol, 0, 255)
            frame[glow_mask, 2] = np.clip(frame[glow_mask,2]+alpha*200*vol, 0, 255)
    elif style == "Minimal Flash":
        c = 255 if vol > 0.65 else int(vol * 40); frame[:] = (c, c, c)
    return frame

# --- MAIN CONTENT FLOW ---
m_col1, m_col2, m_col3 = st.columns([1, 1, 1])

with m_col2:
    # 1. UPLOAD
    uploaded_file = st.file_uploader(
        "Upload your audio or video file", 
        type=["mp3", "wav", "mp4", "mov", "m4a"],
        label_visibility="visible" # Accessibility fix
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.markdown("---")
        
        # 2. SOURCE PREVIEW
        st.markdown("### :material/play_circle: Preview Source")
        if file_type in ['mp4', 'mov']: st.video(uploaded_file)
        else: st.audio(uploaded_file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. CORE SETTINGS
        st.markdown("### :material/palette: Design Studio")
        
        # Style with Preview
        visual_style = st.selectbox("Visual Theme", STYLES, help="Select a visual style for your music.")
        
        preview_path = asset_filename(visual_style)
        if os.path.exists(preview_path):
            st.image(preview_path, use_container_width=True)
        
        # Platform
        platform_list = list(PLATFORMS.keys())
        platform_preset = st.selectbox(
            "Where will you publish?",
            platform_list,
            help="This sets the video orientation (Wide or Vertical)."
        )
        orientation = PLATFORMS[platform_preset]

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4. GENERATE
        generate_btn = st.button("Generate Video", icon=":material/magic_button:")

        if generate_btn:
            output_video_path = f"output_{uuid.uuid4().hex[:8]}.mp4"
            with st.status(":material/rocket_launch: Creating magic...", expanded=True) as status:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix="."+file_type) 
                tfile.write(uploaded_file.read()); temp_input_path = tfile.name; tfile.close()
                try:
                    status.write(":material/analytics: Analyzing audio spectrum...")
                    y, sr, dur, rms, spec = get_audio_features(temp_input_path)
                    
                    rend_dur = 30 if duration_mode == "Preview (30s)" else dur
                    if resolution_mode == "Mobile Low (480p)": W, H = 854, 480
                    elif resolution_mode == "HD (720p)": W, H = 1280, 720
                    else: W, H = 1920, 1080
                    if "Portrait" in orientation: W, H = H, W
                    
                    status.write(f":material/brush: Rendering {int(rend_dur * 24)} frames...")
                    progress_bar = st.progress(0, text="Rendering frames...")
                    
                    def mf(t):
                        # Update progress bar
                        prog = min(t / rend_dur, 1.0)
                        progress_bar.progress(prog, text=f"Rendering: {int(prog*100)}%")
                        return draw_frame(t, visual_style, rms, spec, sr, W, H)
                    
                    clip = VideoClip(mf, duration=rend_dur)
                    audio = AudioFileClip(temp_input_path).subclip(0, rend_dur)
                    clip.set_audio(audio).write_videofile(output_video_path, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast', logger=None)
                    
                    status.update(label=":material/check_circle: Ready!", state="complete", expanded=False)
                    st.balloons()
                
                except MemoryError:
                    st.error(":material/memory: File too large or resolution too high. Try 'Preview' or 'Mobile Low'.")
                except Exception as e:
                    st.error(f":material/error: An error occurred: {e}")
                finally: 
                    if os.path.exists(temp_input_path): os.remove(temp_input_path)

            # 5. RESULTS
            if os.path.exists(output_video_path):
                st.markdown("---")
                st.markdown("### :material/check_circle: Your Masterpiece")
                st.video(output_video_path)
                
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="DOWNLOAD VIDEO",
                        data=f,
                        file_name="visualizer.mp4",
                        mime="video/mp4",
                        icon=":material/download:"
                    )
                
                # SHARE
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### :material/share: Share Your Creation")
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.link_button("YouTube", "https://studio.youtube.com", icon=":material/smart_display:", use_container_width=True)
                s_col2.link_button("TikTok", "https://www.tiktok.com/upload", icon=":material/music_note:", use_container_width=True)
                s_col3.link_button("Instagram", "https://www.instagram.com/", icon=":material/photo_camera:", use_container_width=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info(":material/info: Adjust Resolution and Duration in the sidebar menu.")