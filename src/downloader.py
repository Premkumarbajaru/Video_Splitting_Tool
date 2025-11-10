from pathlib import Path
from .utils import ensure_dir, run_cmd


def download(url: str, out_dir: str) -> Path:
    ensure_dir(out_dir)
    template = str(Path(out_dir) / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", template,
        "--no-warnings",
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "--extractor-retries", "3",
        "--socket-timeout", "30",
        url,
        "--print", "after_move:filepath",
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {res.stderr}")
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    return Path(lines[-1]) if lines else None
