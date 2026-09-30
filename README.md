# Project A.R.G.U.S.

Autonomous Real-time Generative User System — a local-first, multimodal agent
optimized for consumer GPU hardware (target: NVIDIA RTX 3060, 8GB VRAM).

Badges: Status — Sprint 1 Complete

Overview
- Goal: continuous environmental awareness + long-term memory + autonomous tool execution.
- Design constraint: run fully locally with efficient VRAM/CPU use.

What’s in this repo
- `src/` — prototype modules and scripts (audio/vision/agent components).
- `models/` — trained adapters and checkpoints (large files excluded from git).
- `data/` — local metadata and small CSVs used by scripts.
- `whisper/` — local speech model weights (not tracked by git by default).

Quick start
1. Create and activate a virtualenv (Windows PowerShell):

```powershell
python -m venv .ARGUS_venv
.\.ARGUS_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the tactical hearing prototype:

```powershell
cd src
& ..\.ARGUS_venv\Scripts\python.exe argus_hearing.py
```

Architecture (high level)
- Sensory layer: audio (VAD + Whisper), vision (Qwen2-VL adapters).
- Cognitive layer: agentic decision-making and short-term reasoning.
- Memory layer: local vector DB for embeddings and long-term recall.
- Execution layer: safe OS-level action executor (sandboxed by policy rules).

Guiding principles
- Local-first: avoid external APIs for core capabilities.
- Minimal VRAM footprint: offload when idle, use FP16 where safe.
- Modular: clear separation between perception, cognitive logic, and execution.

Development notes
- Large model weights are kept out of git — place adapters in `models/`.
- Use `requirements.txt` to reproduce the Python environment.
- Tests and CI are TODO — see `CONTRIBUTING.md` for contribution guidance.

License
- See `LICENSE` in the repo root.

Contact
- For questions, open an issue or contact the maintainers listed in the project file headers.
