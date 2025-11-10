from pathlib import Path
import os
from .utils import ensure_dir, run_cmd


def separate_vocals(audio_path: str | Path, out_dir: str) -> tuple[Path, Path]:
    audio_path = Path(audio_path).resolve()
    ensure_dir(out_dir)
    # demucs outputs into ./separated/<model>/<track>/vocals.wav etc.
    # Use htdemucs to avoid DiffQ dependency on Windows
    cmd = ["demucs", "-n", "htdemucs", "--two-stems=vocals", str(audio_path)]
    # Prefer sox_io backend and disable torchcodec to avoid extra dependency
    env = {"TORCHAUDIO_USE_BACKEND": "sox_io", "TORCHAUDIO_DISABLE_TORCHCODEC": "1"}
    # Ensure ffmpeg is visible to demucs/torchaudio: if FFMPEG_PATH is set, prepend its dir to PATH
    ff = os.environ.get("FFMPEG_PATH")
    if ff:
        ff_dir = str(Path(ff).parent)
        env["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
    res = run_cmd(cmd, env=env)
    if res.returncode != 0:
        msg = (res.stderr or "").strip()
        if not msg:
            msg = (res.stdout or "").strip()
        raise RuntimeError(f"demucs failed: {msg}")
    # Find latest demucs output
    sep_root = Path("separated")
    latest = max(sep_root.rglob(audio_path.stem), key=lambda p: p.stat().st_mtime)
    vocals = latest / "vocals.wav"
    accomp = latest / "no_vocals.wav"
    # Some demucs variants use 'other.wav' or 'accompaniment.wav'; fallback search
    if not accomp.exists():
        for alt in ("accompaniment.wav", "other.wav", "bass.wav"):
            cand = latest / alt
            if cand.exists():
                accomp = cand
                break
    # Copy or reference into out_dir
    target_vocals = Path(out_dir) / (audio_path.stem + "_vocals.wav")
    target_music = Path(out_dir) / (audio_path.stem + "_accompaniment.wav")
    run_cmd(["ffmpeg", "-y", "-i", str(vocals), str(target_vocals)])
    run_cmd(["ffmpeg", "-y", "-i", str(accomp), str(target_music)])
    return target_vocals, target_music
