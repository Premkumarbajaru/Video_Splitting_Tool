from pathlib import Path
from .utils import ensure_dir, run_cmd


def extract_audio(video_path: str | Path, out_audio_dir: str) -> Path:
    video_path = Path(video_path).resolve()
    ensure_dir(out_audio_dir)
    out_path = Path(out_audio_dir) / (video_path.stem + ".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-q:a",
        "0",
        "-map",
        "a",
        str(out_path),
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {res.stderr}")
    return out_path
