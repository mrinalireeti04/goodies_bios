# FORGE TRACE — Task Intelligence & Timeline Generation

FORGE TRACE is the video-understanding module of the project. It takes raw egocentric factory video recordings and automatically generates a structured, chronological task timeline in JSON format.

---

## 🚀 Quick Start

### 1. Requirements & Installation
Ensure you have Python 3.10+ installed with the following packages:
```powershell
pip install google-genai opencv-python pydantic tenacity python-dotenv pytest
```

### 2. Configure Gemini API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Both standard `AIzaSy...` keys and newer `AQ.Ab...` Google AI Studio authorization keys are supported).*

### 3. Run Pipeline on a Video
```powershell
python run_video.py sample_videos\mytronlabs-000003-013.mp4
```

The output JSON will be generated automatically at:
```text
outputs/<clip_id>_timeline.json
```

---

## 🧠 Approach & Pipeline Architecture

The pipeline processes videos in a 6-stage multi-pass architecture designed for robust action boundary detection and temporal accuracy:

```text
[ Raw Video ]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1. Video Segmentation & Upload                         │
│    • Analyzes video metadata (FPS, duration, codec)    │
│    • Slices into 120s windows with 15s overlap         │
│    • Uploads to Gemini Files API                       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Coarse Action Recognition (Pass 1)                  │
│    • Gemini 3.5 Flash evaluates each video segment     │
│    • Extracts candidates: verb, object, hand,          │
│      start/end timestamps, confidence, description     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Temporal Refinement (Pass 2)                        │
│    • Tightens action start/end boundaries              │
│    • Frame-level sampling (OpenCV) for precise edges   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Normalization & Deduplication                       │
│    • Verb canonicalization ("picks up", "grab" → "pick up")│
│    • Cross-segment merge across the 15s overlap window │
│    • Temporal IoU deduplication (merges duplicate span)│
│    • Low-confidence filtering                          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Multimodal Enrichment (Optional)                    │
│    • Merges external hand-tracking (e.g. MediaPipe /   │
│      RTMPose from the `mediapipe` branch)              │
│    • Fills missing hands/objects & boosts confidence   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6. Task Building & Timeline Synthesis                  │
│    • Assigns chronological IDs: T001, T002, ...        │
│    • Standardizes task names: "<Verb> <Object>"        │
│    • Groups consecutive tasks into macro-phases        │
│    • Validates schema & writes final output JSON       │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
goodies_bios/
├── forge_trace/               # Core task intelligence package
│   ├── action_normalizer.py   # Verb mapping, IoU deduplication, boundary merge
│   ├── action_recognizer.py   # Gemini multimodal segment prompt & parser
│   ├── config.py              # Central pipeline settings (model, thresholds)
│   ├── enrichment.py          # Multimodal merger for hand tracks & object boxes
│   ├── evaluation.py          # Ground-truth matching & IoU evaluation metrics
│   ├── exceptions.py          # Custom error types
│   ├── gemini_client.py       # google-genai SDK wrapper with retry logic
│   ├── logging_config.py      # Standardized logger format
│   ├── pipeline.py            # End-to-end orchestrator
│   ├── schemas.py             # Pydantic schemas (TaskRecord, TaskTimeline)
│   ├── task_builder.py        # Task ID generation and heuristic grouping
│   ├── temporal_refiner.py    # Two-pass boundary refinement
│   ├── validators.py          # Pre-flight and post-flight data validation
│   └── video_utils.py         # OpenCV video metadata & frame extraction
├── outputs/                   # Generated JSON timelines
│   └── MTL-SGF00106-028_timeline.json
├── sample_videos/             # Local video storage (git-ignored)
├── tests/                     # Test suite (109 unit tests)
│   ├── test_action_normalizer.py
│   ├── test_action_recognizer.py
│   ├── test_enrichment.py
│   ├── test_evaluation.py
│   ├── test_schemas.py
│   ├── test_task_builder.py
│   ├── test_validators.py
│   └── test_video_utils.py
├── .env                       # Local secrets (git-ignored)
├── .env.example               # Example env template
├── .gitignore                 # Protects secrets & large video files
├── pyproject.toml             # Project dependencies and configuration
└── run_video.py               # Main CLI runner script
```

---

## 📋 Output JSON Schema

The generated JSON file adheres to the following structure:

```json
{
  "clip_id": "MTL-SGF00106-028",
  "video_path": "sample_videos/MTL-SGF00106-028.mp4",
  "video_duration": 180.0,
  "total_tasks": 17,
  "unique_task_types": 4,
  "unique_task_list": [
    "Align wood piece",
    "Cut wood piece",
    "Fasten wood piece",
    "Mark wood piece"
  ],
  "tasks": [
    {
      "task_id": "T001",
      "task_name": "Align wood piece",
      "verb": "align",
      "object": "wood piece",
      "hand": "both",
      "timeline": {
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0
      },
      "confidence": 0.95,
      "description": "The worker aligns a wooden block inside the frame of a chair.",
      "evidence": "Gemini video analysis (refined)"
    }
  ]
}
```

---

## 🛠️ CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `video_path` | Path to the video file to process (Required) | — |
| `--clip-id` | Custom unique clip identifier | Filename without extension |
| `--output-dir` | Directory to save the output JSON | `outputs` |
| `--api-key` | Gemini API key (optional if set in `.env`) | `GEMINI_API_KEY` |

**Example:**
```powershell
python run_video.py sample_videos\mytronlabs-000003-013.mp4 --output-dir custom_results --clip-id clip_01
```

---

## 🧪 Testing

Run the automated pytest test suite:
```powershell
pytest
```
*Current test suite: **109 tests passing (100%)** covering validators, schemas, video utilities, action normalization, and boundary refining.*
