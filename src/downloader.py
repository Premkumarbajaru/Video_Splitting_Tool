from pathlib import Path
from .utils import ensure_dir, run_cmd


def download(url: str, out_dir: str) -> Path:
    ensure_dir(out_dir)
    # yt-dlp will generate title-based filename with extension
    template = str(Path(out_dir) / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "best",
        "-o",
        template,
        url,
        "--print",
        "after_move:filepath",
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {res.stderr}")
    # take last printed path
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    return Path(lines[-1]) if lines else None
