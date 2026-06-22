# Project A.R.G.U.S.
### Autonomous Real-time Generative User System

![Project Status](https://img.shields.io/badge/Status-Sprint%201%20Complete-green)
![Academic Year](https://img.shields.io/badge/Course-CSE499A%20%2F%20CSE499B-blue)
![Target Hardware](https://img.shields.io/badge/Optimized_For-RTX_3060_(8GB_VRAM)-orange)

---

## 1. Project Abstract & Core Challenge
Project A.R.G.U.S. is an edge-optimized, proactive multimodal AI companion designed to achieve continuous environmental awareness, long-term spatial/temporal memory, and autonomous OS-level tool execution. 

### The Engineering Constraint
Unlike typical cloud-dependent API wrappers, A.R.G.U.S. runs **entirely locally**. The primary software engineering challenge of this 7-month Capstone is optimizing a heavy multi-model pipeline (Vision, Speech-to-Text, Text-to-Speech, Vector Databases) to operate concurrently under strict consumer-grade hardware limits, specifically an **NVIDIA RTX 3060 with 8GB of VRAM**.

---

## 2. System Architecture (Target State)
To prevent VRAM overallocation and race conditions, the project utilizes a decoupled, asynchronous microservice pipeline:
* **Sensory Hearing Engine:** Silero VAD + `faster-whisper` (CTranslate2)
* **Sensory Vision Engine:** Qwen2.5-VL (Vision-Language Model)
* **Cognitive & Execution Layer:** LangChain/LlamaIndex Agentic Framework
* **Memory Vault:** Local Vector Database (ChromaDB / FAISS)
* **Vocal Generation:** Kokoro-82M Text-to-Speech engine

---

## 3. Sprint 1 Deliverables: Tactical Hearing Engine
During the first sprint, the foundation of **Pillar 1 (Sensory Ingestion)** was successfully designed, implemented, and benchmarked.

### Features Implemented:
* **CPU-Based Voice Activity Detection (VAD):** Integrated `webrtcvad` to continually monitor ambient audio streams entirely on the CPU. The GPU remains completely idle during silence, preventing unnecessary VRAM utilization and thermal throttling.
* **Asynchronous Audio Buffering:** Implemented a rolling memory buffer that captures audio frames when human speech is detected and cleanly cuts off after 50 frames (~1 second) of consecutive silence.
* **Optimized GPU Inference Handoff:** Deployed the English-only `base.en` Whisper model through the highly optimized `faster-whisper` CTranslate2 backend. 
* **VRAM Footprint Reduction:** Forced the model execution into `float16` half-precision precision, cutting model VRAM allocation in half while keeping latency remarkably low.
* **Hallucination & Artifact Filtering:** Programmed a programmatic logic gate to catch and discard typical static artifacts (such as single-word hallucinated loops like "You", "Thank you") before passing data to the transcription console.
* **Real-Time Latency Telemetry:** Embedded execution timers to output precise mathematical processing speed (`[Lat: X.XXs]`) alongside localized system timestamps for every spoken sentence.

### Running the Engine
Use the repository virtual environment and the PowerShell call operator to avoid interpreter path issues:
```powershell
cd "C:\Users\Sn1pEx\Desktop\Project_A-R-G-U-S\src"
& "C:\Users\Sn1pEx\Desktop\Project_A-R-G-U-S\.ARGUS_venv\Scripts\python.exe" argus_hearing.py
```

---

## 4. Current Repository Structure
```text
ARGUS-Core/
│
├── src/
│   └── argus_hearing.py      # Core audio streaming, VAD gating, and Whisper inference
│
├── .gitignore                # Protects repository from heavy AI weights (.bin/.safetensors)
└── README.md                 # Project technical documentation