from pathlib import Path
from .utils import ensure_dir, run_cmd


def remove_silence(vocals_path: str | Path, out_audio_dir: str,
                   stop_periods: int = -1, stop_threshold: str = "-50dB", detection: str = "peak") -> Path:
    vocals_path = Path(vocals_path)
    ensure_dir(out_audio_dir)
    out_path = Path(out_audio_dir) / (vocals_path.stem + "_clean.wav")
    filter_arg = f"silenceremove=stop_periods={stop_periods}:stop_threshold={stop_threshold}:detection={detection}"
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_path), "-af", filter_arg, str(out_path)
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg silenceremove failed: {res.stderr}")
    return out_path
