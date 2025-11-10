import argparse
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from src.utils import ensure_dir
from src.downloader import download
from src.extractor import extract_audio
from src.splitter import separate_vocals
from src.transcriber import transcribe
from src.cleaner import remove_silence
from src.segmenter import segment_transcript
from src.clipper import create_clips
from src.merger import find_and_merge_streams


def load_config(cfg_path: Path) -> dict:
    default = {
        "paths": {
            "downloads": "downloads",
            "processed": "processed",
            "audio": "processed/audio",
            "vocals": "processed/vocals",
            "clips": "processed/clips",
            "transcripts": "processed/transcripts",
        },
        "transcription": {"model_size": "small", "compute_type": "int8"},
        "silence_removal": {
            "stop_periods": -1,
            "stop_threshold": "-50dB",
            "detection": "peak",
        },
        "segmentation": {"max_segment_seconds": 60},
    }
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            # shallow merge
            for k, v in loaded.items():
                if isinstance(v, dict) and k in default:
                    default[k].update(v)
                else:
                    default[k] = v
    return default


def main():
    parser = argparse.ArgumentParser(description="Video Splitting Tool")
    parser.add_argument("--url", type=str, help="YouTube or audio/video URL", required=False)
    parser.add_argument("--input_video", type=str, help="Path to a local video file", required=False)
    parser.add_argument("--output", type=str, default="processed", help="Output root directory")
    args = parser.parse_args()

    root = Path(__file__).parent
    cfg = load_config(root / "config.yaml")

    paths = cfg["paths"]
    downloads = ensure_dir(str(root / paths["downloads"]))
    processed = ensure_dir(str(root / paths["processed"]))
    audio_dir = ensure_dir(str(root / paths["audio"]))
    vocals_dir = ensure_dir(str(root / paths["vocals"]))
    clips_dir = ensure_dir(str(root / paths["clips"]))
    transcripts_dir = ensure_dir(str(root / paths["transcripts"]))

    # Step 1: Download or use provided video
    if args.input_video:
        video_path = Path(args.input_video)
        if not video_path.is_absolute():
            video_path = (root / video_path).resolve()
        if not video_path.exists():
            # Try to auto-merge separate audio/video from ./output
            search_dir = root / "output"
            merged = find_and_merge_streams(search_dir, processed, output_name=video_path.name)
            if merged and merged.exists():
                video_path = merged
            else:
                raise SystemExit(f"Input video not found: {video_path}. Also failed to auto-merge from {search_dir}.")
    elif args.url:
        video_path = download(args.url, downloads)
    else:
        raise SystemExit("Provide --url or --input_video")

    # Step 2: Extract audio
    audio_path = extract_audio(video_path, audio_dir)

    # Step 3: Vocal separation (with fallback)
    try:
        vocals_path, _music_path = separate_vocals(audio_path, vocals_dir)
    except Exception as e:
        print(f"Warning: Vocal separation failed ({e}). Proceeding with original audio.")
        vocals_path = audio_path

    # Step 4: Transcription
    tr_cfg = cfg["transcription"]
    transcript = transcribe(vocals_path, tr_cfg.get("model_size", "small"), tr_cfg.get("compute_type", "int8"))

    # Save raw transcript
    transcript_path = Path(transcripts_dir) / f"{video_path.stem}_transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    # Step 5: Silence removal (clean vocals)
    s_cfg = cfg["silence_removal"]
    clean_audio = remove_silence(vocals_path, audio_dir, s_cfg["stop_periods"], s_cfg["stop_threshold"], s_cfg["detection"]) 

    # Step 6: Text segmentation
    seg_cfg = cfg["segmentation"]
    segments = segment_transcript(transcript, seg_cfg.get("max_segment_seconds", 60))

    # Save segments metadata
    segments_path = Path(transcripts_dir) / f"{video_path.stem}_segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # Step 7: Clip creation
    clip_paths = create_clips(video_path, segments, clips_dir)

    print("Outputs:")
    print(f"  Transcript: {transcript_path}")
    print(f"  Segments:   {segments_path}")
    print(f"  Clean audio:{clean_audio}")
    for cp in clip_paths:
        print(f"  Clip:       {cp}")


if __name__ == "__main__":
    main()
