import os
import subprocess
import shutil
import sys
import re
from pathlib import Path

def sanitize_path(path: str | Path) -> Path:
    """Sanitize a path by replacing problematic Unicode chars and handling Windows restrictions."""
    path = Path(path)
    
    # Map of Unicode chars to ASCII replacements
    repl = {
        '｜': '_',  # FULLWIDTH VERTICAL LINE
        '|': '_',   # regular vertical line
        '"': '_',   # curly quotes
        '"': '_',
        ''': '_',
        ''': '_',
        '–': '-',   # en-dash
        '—': '-',   # em-dash
        '…': '...',
        ' ': '_',   # spaces to underscores
    }
    
    # Clean the stem (filename without extension)
    stem = path.stem
    for k, v in repl.items():
        stem = stem.replace(k, v)
    
    # Replace remaining non-ASCII chars with underscore
    stem = ''.join(c if ord(c) < 128 and c.isprintable() else '_' for c in stem)
    # Collapse multiple underscores
    stem = re.sub(r'_+', '_', stem)
    # Remove leading/trailing underscores
    stem = stem.strip('_')
    
    return path.parent / f"{stem}{path.suffix}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def run_cmd(cmd: list, cwd: str | None = None, env: dict | None = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command with proper encoding handling for Windows."""
    if not cmd:
        raise ValueError("Empty command")
    exe = cmd[0]
    
    # Base environment with UTF-8 encoding
    proc_env = os.environ.copy()
    proc_env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONLEGACYWINDOWSSTDIO": "1"
    })
    if env:
        proc_env.update(env)

    # ffmpeg: resolve full path
    if exe == "ffmpeg":
        # 1) Env var override
        ff_env = os.environ.get("FFMPEG_PATH")
        candidates = []
        if ff_env:
            candidates.append(ff_env)
        # 2) PATH
        ff_path = shutil.which("ffmpeg")
        if ff_path:
            candidates.append(ff_path)
        # 3) Common Windows locations
        common = [
            r"C:\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files\\Gyan\\ffmpeg\\bin\\ffmpeg.exe",
        ]
        candidates.extend(common)
        ff = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not ff:
            raise FileNotFoundError(
                "FFmpeg not found. Set FFMPEG_PATH env var to ffmpeg.exe or add ffmpeg to PATH."
            )
        cmd = [ff] + cmd[1:]

    # demucs: prefer python -m demucs to avoid PATH issues
    if exe == "demucs":
        cmd = [sys.executable, "-m", "demucs"] + cmd[1:]

    # yt-dlp: prefer python -m yt_dlp
    if exe in ("yt-dlp", "yt_dlp"):
        cmd = [sys.executable, "-m", "yt_dlp"] + cmd[1:]

    # Merge environment
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    kwargs = {
        "cwd": cwd,
        "env": proc_env
    }
    
    if capture_output:
        # Use pipes with UTF-8 encoding for output capture
        import io
        import subprocess
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs
        )
        
        # Read output using UTF-8
        try:
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(
                cmd,
                process.returncode,
                stdout.decode('utf-8', errors='replace'),
                stderr.decode('utf-8', errors='replace')
            )
        except Exception as e:
            process.kill()
            raise RuntimeError(f"Command failed: {e}")
    else:
        # Run without output capture
        return subprocess.run(cmd, **kwargs)
