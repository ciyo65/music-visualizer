import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import uuid
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
        
        # Get frequency column for this frame
        if frame_idx < spec_norm.shape[1]:
            # spec_norm shape is (1025, num_frames)
            # We want to ignore the very high frequencies (top of the spectrogram) 
            # as they are often empty or noise. Let's use the first 200 bins.
            freq_col = spec_norm[:200, frame_idx] 
        else:
            freq_col = np.zeros(200)

        # Resample frequency bins to number of bars
        # We can just split the array into 'num_bars' chunks and average them
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
            
            # Gradient color: Low freqs (left) = Blue/Green, High freqs (right) = Red/Yellow
            # Simple static green for now as requested, or dynamic:
            # color = [int(i/num_bars * 255), 255 - int(i/num_bars * 255), 150]
            frame[y1:H, x1:x2] = [0, 255, 100]

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