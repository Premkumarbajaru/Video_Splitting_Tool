import os
from pathlib import Path
from typing import List, Dict
from .utils import ensure_dir, run_cmd, sanitize_path


def create_clips(video_path: str | Path, segments: List[Dict], clips_dir: str) -> list[Path]:
    """Create clips from video_path for each segment, preserving all streams including audio."""
    video_path = Path(video_path)
    ensure_dir(clips_dir)
    outputs = []
    
    # Sanitize input path and check it exists
    safe_input = sanitize_path(video_path)
    if not video_path.exists() and safe_input.exists():
        video_path = safe_input
    elif not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # First verify input video has audio
    print(f"Analyzing input video: {video_path}")
    probe_cmd = ["ffmpeg", "-i", str(video_path)]
    res = run_cmd(probe_cmd)
    if "Audio" not in (res.stderr or ""):
        print("⚠️ Warning: No audio streams detected in input video")

    # Windows long path helper
    def _lp(p: Path) -> str:
        sp = str(p)
        if os.name == 'nt':
            sp = str(p.resolve())  # Normalize path
            if not sp.startswith("\\\\?\\"):
                return "\\\\?\\" + sp
        return sp

    for s in segments:
        out = Path(clips_dir) / f"clip_{s['segment_id']}.mp4"
        
        # Remove existing file if it exists to avoid permission issues
        if out.exists():
            try:
                out.unlink()
            except:
                pass
        
        start = float(s["start"])
        end = float(s["end"])
        duration = max(0, end - start)

        # Use -ss before -i for fast seeking
        # Use -map 0 to ensure all streams (video+audio) are copied
        print(f"Creating clip {s['segment_id']} ({duration:.1f}s)")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),  # Use raw path, let ffmpeg handle it
            "-t", str(duration),
            "-map", "0",        # Copy ALL streams
            "-c:v", "copy",     # Copy video codec
            "-c:a", "copy",     # Copy audio codec
            str(out)            # Raw output path
        ]
        res = run_cmd(cmd)
        if res.returncode != 0:
            print(f"❌ Error creating clip {s['segment_id']}:")
            print(res.stderr)
            raise RuntimeError(f"ffmpeg clip failed for segment {s['segment_id']}")
            
        # Verify the output has audio
        verify_cmd = ["ffmpeg", "-i", str(out)]
        verify_res = run_cmd(verify_cmd)
        if "Audio" not in (verify_res.stderr or ""):
            print(f"⚠️ Warning: No audio detected in output clip {s['segment_id']}")
            
        outputs.append(out)
    return outputs
