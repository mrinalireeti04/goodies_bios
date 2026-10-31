"""Test script to run the complete FORGE TRACE Task Timeline pipeline on a video."""

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

from forge_trace.exceptions import ForgeTraceError
from forge_trace.logging_config import get_logger
from forge_trace.pipeline import process_video_to_timeline

logger = get_logger("run_video")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FORGE TRACE Task Timeline pipeline on a video."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Optional unique clip identifier. Defaults to video file stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save the formatted JSON output (default: outputs).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--tools-path",
        type=str,
        default=None,
        help="Optional path to PS2 tool-hold events JSON file.",
    )
    parser.add_argument(
        "--single-request",
        action="store_true",
        help="Process video in a single request (recommended for clips <= 3-4 mins).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    clip_id = args.clip_id or video_path.stem
    if args.api_key:
        import os
        os.environ["GEMINI_API_KEY"] = args.api_key

    tool_events = None
    if args.tools_path:
        tp = Path(args.tools_path)
        if tp.exists():
            with open(tp, "r", encoding="utf-8") as f:
                tool_events = json.load(f)
                if isinstance(tool_events, dict) and "tools" in tool_events:
                    tool_events = tool_events["tools"]
        else:
            print(f"Warning: Tools file not found: {tp}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{clip_id}_timeline.json"

    print(f"Processing video: {video_path}")
    print(f"Clip ID: {clip_id}")

    from forge_trace.config import load_config
    cfg = load_config(single_request_mode=args.single_request)

    try:
        timeline = process_video_to_timeline(
            video_path=str(video_path),
            clip_id=clip_id,
            object_events=None,
            hand_tracks=None,
            interaction_events=None,
            tool_events=tool_events,
            config=cfg,
        )
    except ForgeTraceError as err:
        print(f"Pipeline error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error during processing: {err}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Total Identified Tasks: {timeline.total_tasks}")
    print("=" * 60)

    for task in timeline.tasks:
        print(
            f"ID: {task.task_id} | Name: {task.task_name} | "
            f"Start: {task.timeline.start_s:.2f}s | End: {task.timeline.end_s:.2f}s | "
            f"Duration: {task.timeline.duration_s:.2f}s | "
            f"Object: {task.object} | Hand: {task.hand}"
        )

    # Save formatted JSON output
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(timeline.model_dump_json(indent=2))
        print("\n" + "=" * 60)
        print(f"Output successfully saved to: {output_file.resolve()}")
        print("=" * 60)
    except Exception as err:
        print(f"Error writing output JSON: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
