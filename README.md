# Exercise Form Grader

Computer vision project for the CSC_51073 (Computer Vision) course at Polytechnique Montréal.
Given a video of someone performing a gym exercise, the pipeline extracts body pose landmarks,
segments the video into repetitions, aligns each repetition against a reference performance
using Dynamic Time Warping, and scores how closely the joint angles match the reference.

## Pipeline

```
video (.mp4)
    │
    ▼
landmark_extraction.py   MediaPipe Pose → per-frame 33 landmarks (x, y, z),
    │                    gaps interpolated across missing frames
    ▼
grader.py            rep segmentation (peak detection on a tracked landmark)
    │                 + DTW alignment against ref/<exercise>.csv
    ▼
metric.py             per-frame joint-angle similarity → per-repetition score
    │
    ▼
display.py (optional)   video overlay: skeleton, rep counter, live score
```

`ExerciseClasses.py` holds the per-exercise configuration (which landmark/axis drives rep
detection, and the angle weights / scoring calibration used by `metric.py`).

`DataClasses.py` and `Models.py` implement a separate, experimental exercise-classifier
(pose → exercise type) built on top of a TensorFlow model. It is not wired into the grading
CLI below; it was used to explore automatic exercise recognition.

## Installation

Requires Python 3.11. MediaPipe's legacy `solutions` API (used here for pose landmarks) was
removed in MediaPipe 1.0, so the version is pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For running the tests, use `requirements-dev.txt` instead (adds `pytest`).

## Usage

```bash
python main.py <path/to/video.mp4> <exercise> [--display]
```

`--display` opens a window with the video, the tracked skeleton, the rep counter, the live
per-repetition score, and the exercise name/icon (`icons/<exercise>.png`) if available
(requires a GUI environment; skip it when running headless).

Supported exercises (see `ref/*.csv` for the reference performance of each): `squat`,
`push-up`, `leg extension`, `barbell biceps curl`.

Example:

```bash
python main.py "data/data-btc/push-up/push-up_0.mp4" push-up
```

```
Detected 4 repetition(s) for 'push-up':
  rep 1: 1.00
  rep 2: 1.00
  rep 3: 1.00
  rep 4: 0.77
Average score: 0.94
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover the pure scoring logic (`metric.py`, `ExerciseClasses.py`); they don't require a
video or MediaPipe. The end-to-end pipeline (`main.py`) has been manually verified against the
sample videos in `data/`.

## Project structure

```
main.py                 CLI entrypoint
landmark_extraction.py  MediaPipe pose extraction + interpolation (used by main.py)
pose_detection.py       Standalone pose-extraction utilities/experiments
grader.py               Rep segmentation + DTW alignment
metric.py               Joint-angle scoring
ExerciseClasses.py      Per-exercise configuration
display.py              Optional video overlay
DataClasses.py, Models.py   Experimental exercise classifier (not used by main.py)
ref/                     Reference performance per exercise, used by grader.py
icons/                   Per-exercise icon shown by display.py's overlay
tests/                   Unit tests (pytest)
```

`data/` (gitignored) holds the raw/processed videos and datasets used during development and
is not part of the repository.

## Known limitations

- `DataClasses.py` / `Models.py` (exercise classifier) are exploratory and not covered by
  automated tests.
