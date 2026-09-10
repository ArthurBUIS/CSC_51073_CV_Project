"""CLI entrypoint: grade exercise form in a video against a reference performance.

Usage: python main.py <video> <exercise> [--display]
"""

import argparse

from ExerciseClasses import EXERCISES
import grader
import landmark_extraction as le


def run(video_path: str, exercise: str, display: bool) -> None:
    """Extract landmarks from video_path, grade each repetition against
    ref/<exercise>.csv, print the per-repetition scores, and optionally
    replay the video with a live overlay (see display.py)."""
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise '{exercise}'. Available: {', '.join(sorted(EXERCISES))}")

    landmarks, df = le.pipe_extract_landmark(video_path)
    rep_starts, sim_list = grader.compute_repgrade(df, exercise)

    print(f"Detected {len(sim_list)} repetition(s) for '{exercise}':")
    for i, score in enumerate(sim_list, start=1):
        print(f"  rep {i}: {score:.2f}")
    print(f"Average score: {sum(sim_list) / len(sim_list):.2f}")

    if display:
        import display as disp
        disp.play_video_with_landmarks_and_reps(video_path, landmarks, rep_starts, sim_list, exercise)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade exercise form in a video by comparing it to a reference performance."
    )
    parser.add_argument("video", help="Path to the exercise video to grade.")
    parser.add_argument("exercise", choices=sorted(EXERCISES), help="Exercise to grade against.")
    parser.add_argument(
        "--display", action="store_true",
        help="Show the video with landmarks, rep count and live score overlaid.",
    )
    args = parser.parse_args()

    run(args.video, args.exercise, args.display)


if __name__ == "__main__":
    main()
