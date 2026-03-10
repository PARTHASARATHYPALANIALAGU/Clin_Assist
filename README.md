# ClinAssist

**Speech-Driven Structured Clinical Intake & Triage System**

ClinAssist is an AI-powered medical intake assistant designed to automate the process of gathering structured symptom information through natural voice and text interactions. It focuses on **triage rather than diagnosis**, ensuring that critical cases are identified with high urgency using deterministic safety logic.

---

## Features

- **Voice-Native Interaction**: Hands-free clinical intake using frontend VAD (Voice Activity Detection) and high-accuracy STT (Speech-to-Text).
- **Structured Data Extraction**: Automatically extracts 9 core clinical attributes from natural conversation using LLMs (Llama 3.1).
- **Deterministic Risk Engine**: A rule-based safety layer that categorizes patient risk (Critical, High, Moderate, Low) based on severity and "Panic Keywords".
- **Real-time Latency Telemetry**: Tracks performance metrics for STT, LLM, and TTS to ensure a responsive user experience.
- **Local TTS Synthesis**: Uses Piper (ONNX) for sub-300ms speech synthesis with zero cloud dependency for audio generation.
- **Health Consult Tab**: A general health Q&A interface that bridges the gap between intake and patient education.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11+, FastAPI, SQLite |
| **STT (Ear)** | OpenAI Whisper-1 (via Nexus API) |
| **LLM (Brain)** | Llama 3.1 8B (Groq) |
| **TTS (Voice)**| Piper (Local ONNX) |
| **Frontend** | Vanilla JS, HTML5, CSS3 |

---

## Quick Start

### 1. Prerequisites
- Python 3.11 or higher
- [FFmpeg](https://ffmpeg.org/) (for audio processing)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/clinassist.git
cd clinassist

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
NEXUS_API_KEY=your_nexus_api_key
NEXUS_BASE_URL=your_nexus_base_url
GROQ_API_KEY=your_groq_api_key  
```

### 4. Running the App
```bash
uvicorn main:app --reload
```
Visit `http://localhost:8000` to start the intake.

---

## Project Structure

- `main.py`: FastAPI entry point and API endpoints.
- `intake.py`: The State Machine managing the conversation flow.
- `risk.py`: Deterministic rule engine for clinical safety.
- `llm.py`: Integration with Groq/Nexus for symptom extraction.
- `stt.py` & `tts.py`: Speech processing modules.
- `database.py`: SQLite persistence for sessions and telemetry.
- `static/`: Frontend assets (HTML, CSS, JS).

---

## Safety & Disclaimer

ClinAssist is an academic demonstration project. **It does not provide medical diagnosis, treatment advice, or prescriptions.** It is designed solely as a structured data collection tool to assist healthcare providers in triage workflows.

---
