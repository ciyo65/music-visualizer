# How to Run the Professional Music Visualizer

This project now uses a **Full-Stack Architecture**:
- **Frontend:** Next.js (React + TypeScript + Tailwind CSS)
- **Backend:** FastAPI (Python + MoviePy + Librosa)

---

## 1. Start the Backend (Python)
The backend handles the heavy audio analysis and high-quality video rendering.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*The backend will be running at `http://localhost:8000`*

---

## 2. Start the Frontend (Next.js)
The frontend provides the beautiful Pro UI and live canvas previews.

```bash
cd frontend
npm install
npm run dev
```
*Open [http://localhost:3003](http://localhost:3003) in your browser.*

---

## 🛠 Features
- **Live Preview:** See your music react in the browser before you render.
- **High-Quality Export:** Unlike browser-based tools, this uses a robust Python engine for frame-perfect MP4s.
- **Social Ready:** Presets for YouTube (16:9) and TikTok/Reels (9:16).
