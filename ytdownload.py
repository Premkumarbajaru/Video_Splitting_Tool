import subprocess
import os
import re
import glob
import json
from pathlib import Path
import sys
import shutil

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# Ensure ffmpeg is discoverable if user provided FFMPEG_PATH
_ffenv = os.environ.get("FFMPEG_PATH")
if _ffenv:
    p = Path(_ffenv)
    try:
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        elif p.is_file():
            os.environ["PATH"] = str(p.parent) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def extract_video_url(playlist_url):
    """Extracts the single video URL from a playlist URL."""
    match = re.search(r"v=([a-zA-Z0-9_-]+)", playlist_url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return playlist_url


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _find_nearby_file(folder: str | Path, base_stem: str, exts: list[str]) -> str | None:
    folder = Path(folder)
    norm_target = _normalize_name(base_stem)
    exts_l = {e.lower() for e in exts}
    best = None
    for p in folder.glob('*'):
        if not p.is_file():
            continue
        ext = p.suffix.lstrip('.').lower()
        if exts and ext not in exts_l:
            continue
        stem_norm = _normalize_name(p.stem)
        if stem_norm == norm_target or stem_norm.startswith(norm_target):
            if best is None or p.stat().st_size > best.stat().st_size:
                best = p
    return str(best) if best else None


def _latest_file(folder: str | Path, exts: list[str]) -> str | None:
    folder = Path(folder)
    exts_l = {e.lower() for e in exts}
    cands = [p for p in folder.glob('*') if p.is_file() and p.suffix.lstrip('.').lower() in exts_l]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def _fmt_ts(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_cmd(cmd: list, cwd: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    # Build environment; ensure subprocess output is read as bytes and decoded safely
    proc_env = os.environ.copy()
    # prefer UTF-8 for child processes where possible
    proc_env.setdefault("PYTHONIOENCODING", "utf-8")
    proc_env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "1")
    if env:
        proc_env.update(env)

    def _resolve_ffmpeg() -> str | None:
        ff = shutil.which("ffmpeg")
        if ff:
            return ff
        ff_env = os.environ.get("FFMPEG_PATH")
        if ff_env:
            p = Path(ff_env)
            if p.is_file():
                return str(p)
            if p.is_dir():
                cand = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                if cand.exists():
                    return str(cand)
        # try common Windows default
        if os.name == "nt":
            common = Path("C:/ffmpeg/bin/ffmpeg.exe")
            if common.exists():
                return str(common)
        return None

    # Resolve ffmpeg executable if requested
    try:
        if cmd and isinstance(cmd, list) and cmd[0] == "ffmpeg":
            exe = _resolve_ffmpeg()
            if not exe:
                raise FileNotFoundError("ffmpeg executable not found")
            cmd = [exe] + cmd[1:]
            proc_env["PATH"] = str(Path(exe).parent) + os.pathsep + proc_env.get("PATH", "")

        # Run process and capture raw bytes; decode with utf-8 and replace errors
        proc = subprocess.Popen(cmd, cwd=cwd, env=proc_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_bytes, err_bytes = proc.communicate()
        try:
            stdout = out_bytes.decode("utf-8") if out_bytes is not None else ""
        except Exception:
            stdout = out_bytes.decode("utf-8", errors="replace") if out_bytes is not None else ""
        try:
            stderr = err_bytes.decode("utf-8") if err_bytes is not None else ""
        except Exception:
            stderr = err_bytes.decode("utf-8", errors="replace") if err_bytes is not None else ""
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except FileNotFoundError as e:
        if cmd and isinstance(cmd, list) and cmd[0] == "ffmpeg":
            raise SystemExit(
                "FFmpeg not found. Install FFmpeg or set FFMPEG_PATH to your ffmpeg executable.\n"
                "Windows options:\n"
                " - winget install Gyan.FFmpeg\n"
                " - Or download a static build from https://www.gyan.dev/ffmpeg/builds/ and add the bin folder to PATH\n"
                " - You can also set an environment variable FFMPEG_PATH pointing to ffmpeg.exe"
            ) from e
        raise


def require_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff:
        p = Path(env_ff)
        if p.is_file():
            return str(p)
        if p.is_dir():
            cand = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if cand.exists():
                return str(cand)
    if os.name == "nt":
        common = Path("C:/ffmpeg/bin/ffmpeg.exe")
        if common.exists():
            return str(common)
    raise SystemExit(
        "FFmpeg not found. Install FFmpeg or set FFMPEG_PATH to your ffmpeg executable.\n"
        "Windows options:\n"
        " - winget install Gyan.FFmpeg\n"
        " - Or download a static build from https://www.gyan.dev/ffmpeg/builds/ and add the bin folder to PATH\n"
        " - You can also set an environment variable FFMPEG_PATH pointing to ffmpeg.exe"
    )


def convert_video(input_file, target_format):
    """Converts the video to the specified format using H.264 for video and AAC for audio."""
    input_path = Path(input_file).resolve()
    
    # If input doesn't exist, try to find it
    if not input_path.exists():
        output_dir = Path("output")
        candidates = list(output_dir.glob("*.[mM][pP]4"))
        if candidates:
            # Take the most recently modified file
            input_path = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent download: {input_path}")
        else:
            print("❌ No video files found in output directory")
            return None
            
    # Create sanitized output filename
    base_name = input_path.stem.strip()
    output_path = input_path.parent / f"{base_name}.{target_format}"
    
    # If input already matches target format, skip
    if input_path.suffix.lower().lstrip(".") == target_format.lower():
        print("ℹ️ Input is already in target format; skipping conversion.")
        return str(input_path)

    ff = require_ffmpeg()
    
    print(f"Converting video:\n  From: {input_path}\n  To: {output_path}")

    # Common ffmpeg arguments for both formats
    common_args = [
        "-c:v", "libx264",     # Video codec
        "-c:a", "aac",         # Audio codec
        "-b:v", "1000k",       # Video bitrate
        "-b:a", "128k",        # Audio bitrate
        "-map", "0:v?",        # Map video streams if present
        "-map", "0:a?"         # Map audio streams if present
    ]
    
    if target_format == "mp4":
        # Additional MP4-specific options
        command = [
            ff, "-i", str(input_path),
            *common_args,
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(output_path)
        ]
    elif target_format == "mkv":
        command = [
            ff, "-i", str(input_path),
            *common_args,
            str(output_path)
        ]
    else:
        print("❌ Unsupported format. No conversion performed.")
        return None

    # Print the exact command being run
    print("Running ffmpeg command:")
    print(" ".join(command))
    
    res = run_cmd(command)
    if res.returncode != 0 or not output_path.exists():
        print("❌ Conversion failed:")
        if res.stderr:
            print(res.stderr.strip())
        return None
        
    print("✅ Conversion successful")
    return str(output_path)


def download_media(video_url, file_type, quality, is_playlist):
    """Downloads media (MP3 or MP4) and returns the downloaded file path. Also offers conversion afterward for videos."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Use ASCII-only filename template with sanitized characters
    filename = os.path.join(output_dir, "%(title).200B.%(ext)s")
    
    # Add options to restrict to ASCII filenames
    restrictfilenames = ["--restrict-filenames", "--windows-filenames"]

    audio_quality_map = {"low": "50K", "medium": "128K", "high": "192K"}
    video_quality_map = {"low": "360", "medium": "720", "high": "1080"}

    command = [sys.executable, "-m", "yt_dlp"] + restrictfilenames + ["-o", filename]

    if file_type == "mp3":
        bitrate = audio_quality_map.get(quality, "128K")
        command += ["-x", "--audio-format", "mp3", "--audio-quality", bitrate]
    elif file_type == "mp4":
        resolution = video_quality_map.get(quality, "720")
        command += [
            "-f", f"bv*[height<={resolution}]+ba/best",
            "--merge-output-format", "mp4"
        ]

    if not is_playlist:
        command.append("--no-playlist")

    # Use yt-dlp's --print after_move:filepath to capture the output filename
    command += ["--print", "after_move:filepath"]
    command.append(video_url)
    result = subprocess.run(command, capture_output=True, text=True)

    # Try to determine output filepath from yt-dlp output
    output_lines = result.stdout.strip().splitlines()
    downloaded_path = output_lines[-1].strip() if output_lines else None

    def _sanitize_path_str(p: str | None) -> str | None:
        if not p:
            return p
        # trim trailing spaces and spaces before extension dot
        p2 = re.sub(r"\s+\.(\w+)$", r".\1", p.strip())
        return p2
    downloaded_path = _sanitize_path_str(downloaded_path)

    # If the reported path does not exist, try to locate it in output_dir
    if downloaded_path and not os.path.isfile(downloaded_path):
        folder = os.path.dirname(downloaded_path) or output_dir
        base = os.path.splitext(os.path.basename(downloaded_path))[0]
        found = _find_nearby_file(folder, base, exts=["mp4", "mkv", "webm", "m4a", "mp3"])
        if found:
            downloaded_path = found

    # Conversion step for video files
    if file_type == "mp4" and downloaded_path:
        input_file = downloaded_path
        # Offer conversion only if ffmpeg is available (robust detection)
        ff_ok = None
        try:
            ff_ok = require_ffmpeg()
        except SystemExit:
            ff_ok = None
        if ff_ok:
            convert_choice = input("Do you want to convert the file to another format? (yes/no): ").strip().lower()
            if convert_choice == "yes":
                target_format = input("Enter target format (mp4/mkv): ").strip().lower()
                output_file = convert_video(input_file, target_format)
                if output_file and os.path.abspath(output_file) != os.path.abspath(input_file):
                    print(f"✅ Converted video saved as: {output_file}")
                    downloaded_path = _sanitize_path_str(output_file)
        else:
            print("ℹ️ FFmpeg not found; skipping optional conversion step.")
    # If file type is mp3, no further conversion is applied.
    return downloaded_path


# ---------------------- Pipeline helpers ----------------------

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
        "silence_removal": {"stop_periods": -1, "stop_threshold": "-50dB", "detection": "peak"},
        "segmentation": {"max_segment_seconds": 60},
    }
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            for k, v in loaded.items():
                if isinstance(v, dict) and k in default:
                    default[k].update(v)
                else:
                    default[k] = v
    return default


def extract_audio(video_path: str | Path, out_audio_dir: str) -> Path:
    video_path = Path(video_path).resolve()
    ensure_dir(out_audio_dir)
    out_path = Path(out_audio_dir) / (video_path.stem + ".wav")
    # Windows long path support for ffmpeg
    def _lp(p: Path) -> str:
        s = str(p)
        if os.name == 'nt' and not s.startswith("\\\\?\\"):
            return "\\\\?\\" + s
        return s
    if not video_path.exists():
        # Try to auto-merge from ./output if missing
        merged = try_auto_merge_from_output(Path(__file__).parent, desired_name=video_path.name)
        if merged:
            video_path = merged.resolve()
        else:
            raise RuntimeError(f"Input video not found: {video_path}")
    # Probe to see if input has an audio stream
    probe = run_cmd(["ffmpeg", "-hide_banner", "-i", _lp(video_path)])
    has_audio = "Audio:" in (probe.stderr or "")
    if not has_audio:
        # Attempt auto-merge of separate streams from ./output, then retry
        merged = try_auto_merge_from_output(Path(__file__).parent, desired_name=video_path.name)
        if merged and merged.exists():
            video_path = merged.resolve()
            probe2 = run_cmd(["ffmpeg", "-hide_banner", "-i", _lp(video_path)])
            has_audio = "Audio:" in (probe2.stderr or "")
    if not has_audio:
        raise RuntimeError(f"No audio stream found in input and auto-merge failed: {video_path}")

    # Extract audio: drop video (-vn) and let ffmpeg pick default WAV encoding
    cmd = [
        "ffmpeg", "-y", "-i", _lp(video_path), "-vn", _lp(out_path)
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {res.stderr}")
    return out_path


# ---------------------- Local merge helper ----------------------

def _base_key(p: Path) -> str:
    name = p.name
    idx = name.rfind('.f')
    return name[:idx] if idx != -1 else p.stem


def try_auto_merge_from_output(root: Path, desired_name: str | None = None) -> Path | None:
    """Merge best video-only .mp4 and audio-only .webm from ./output into ./processed/<desired_name or merged>.mp4"""
    out_dir = root / "output"
    proc_dir = root / "processed"
    ensure_dir(str(proc_dir))
    videos = list(out_dir.glob('*.mp4'))
    audios = list(out_dir.glob('*.webm'))
    if not videos or not audios:
        return None
    vmap, amap = {}, {}
    for v in videos:
        key = _base_key(v)
        if key not in vmap or v.stat().st_size > vmap[key].stat().st_size:
            vmap[key] = v
    for a in audios:
        key = _base_key(a)
        if key not in amap or a.stat().st_size > amap[key].stat().st_size:
            amap[key] = a
    for key, v in vmap.items():
        if key in amap:
            a = amap[key]
            out_name = desired_name or (key + "_merged.mp4")
            safe = out_name.replace('|', '_')
            merged = proc_dir / safe
            cmd = [
                'ffmpeg', '-y',
                '-i', str(v),
                '-i', str(a),
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest', str(merged)
            ]
            res = run_cmd(cmd)
            if res.returncode == 0 and merged.exists():
                return merged
    return None


def _sanitize_filename(s: str) -> str:
    """Replace problematic Unicode chars with ASCII equivalents."""
    # Map of Unicode chars to ASCII replacements
    repl = {
        '｜': '|',  # FULLWIDTH VERTICAL LINE
        '"': '"',   # curly quotes
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',   # en-dash
        '—': '-',   # em-dash
        '…': '...'
    }
    # Replace each char with its ASCII equivalent
    for k, v in repl.items():
        s = s.replace(k, v)
    # Replace any other non-ASCII chars with underscore
    s = ''.join(c if ord(c) < 128 else '_' for c in s)
    return s

def separate_vocals(audio_path: str | Path, out_dir: str) -> tuple[Path, Path]:
    audio_path = Path(audio_path).resolve()
    # Sanitize the stem (filename without extension) to avoid Unicode issues
    if any(ord(c) >= 128 for c in audio_path.stem):
        new_path = audio_path.parent / (_sanitize_filename(audio_path.stem) + audio_path.suffix)
        if audio_path.exists():
            audio_path.rename(new_path)
            audio_path = new_path
    
    ensure_dir(out_dir)
    # Use htdemucs and two-stems=vocals to avoid DiffQ and reduce outputs
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals", str(audio_path)]
    # Set both PYTHONIOENCODING and PYTHONLEGACYWINDOWSSTDIO to handle output encoding
    env = {
        # Force torchaudio to use sox_io and not torchcodec
        "TORCHAUDIO_USE_BACKEND": "sox_io",
        "TORCH_AUDIO_BACKEND": "sox_io",
        "TORCHAUDIO_DISABLE_TORCHCODEC": "1",
        "USE_TORCHCODEC": "0",
        "TORCHCODEC_DISABLE": "1",
        "TORCHCODEC_ENABLED": "0",
        # Windows unicode/stdio safety
        "PYTHONIOENCODING": "utf-8",
        "PYTHONLEGACYWINDOWSSTDIO": "1"
    }
    # Ensure demucs/python prints unicode to stdout/stderr correctly on Windows consoles
    # This prevents demucs from crashing with a UnicodeEncodeError when it prints characters
    # that aren't representable in the default cp1252 encoding.
    env["PYTHONIOENCODING"] = "utf-8"
    ff = os.environ.get("FFMPEG_PATH")
    if ff:
        p = Path(ff)
        ff_dir = p.parent if p.is_file() else p
        env["PATH"] = str(ff_dir) + os.pathsep + os.environ.get("PATH", "")
    res = run_cmd(cmd, env=env)
    if res.returncode != 0:
        msg = (res.stderr or "").strip() or (res.stdout or "").strip()
        raise RuntimeError(f"demucs failed: {msg}")
    sep_root = Path("separated")
    candidates = list(sep_root.rglob(audio_path.stem))
    if not candidates:
        raise RuntimeError("demucs output not found")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    vocals = latest / "vocals.wav"
    accomp = latest / "no_vocals.wav"
    if not accomp.exists():
        for alt in ("accompaniment.wav", "other.wav", "bass.wav"):
            cand = latest / alt
            if cand.exists():
                accomp = cand
                break
    target_vocals = Path(out_dir) / (audio_path.stem + "_vocals.wav")
    target_music = Path(out_dir) / (audio_path.stem + "_accompaniment.wav")
    run_cmd(["ffmpeg", "-y", "-i", str(vocals), str(target_vocals)])
    run_cmd(["ffmpeg", "-y", "-i", str(accomp), str(target_music)])
    return target_vocals, target_music


def transcribe_audio(audio_path: str | Path, model_size: str = "small", compute_type: str = "int8"):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("faster-whisper not installed. Please install requirements.txt") from e
    model = WhisperModel(model_size, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), vad_filter=True)
    out = []
    for i, seg in enumerate(segments):
        out.append({"id": i + 1, "start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return out


def remove_silence(vocals_path: str | Path, out_audio_dir: str, stop_periods: int = -1, stop_threshold: str = "-50dB", detection: str = "peak") -> Path:
    vocals_path = Path(vocals_path)
    ensure_dir(out_audio_dir)
    out_path = Path(out_audio_dir) / (vocals_path.stem + "_clean.wav")
    filter_arg = f"silenceremove=stop_periods={stop_periods}:stop_threshold={stop_threshold}:detection={detection}"
    cmd = ["ffmpeg", "-y", "-i", str(vocals_path), "-af", filter_arg, str(out_path)]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg silenceremove failed: {res.stderr}")
    return out_path


def segment_transcript(segments: list[dict], max_segment_seconds: int = 60) -> list[dict]:
    grouped = []
    current = {"segment_id": 1, "start": None, "end": None, "items": [], "topic": None}
    sid = 1
    acc_start = None
    for s in segments:
        if acc_start is None:
            acc_start = s["start"]
            current["start"] = s["start"]
        current["items"].append(s)
        current["end"] = s["end"]
        if (current["end"] - acc_start) >= max_segment_seconds:
            grouped.append({
                "segment_id": sid,
                "start": current["start"],
                "end": current["end"],
                "text": " ".join(item["text"] for item in current["items"]).strip(),
                "topic": current["topic"] or f"Segment {sid}",
            })
            sid += 1
            current = {"segment_id": sid, "start": None, "end": None, "items": [], "topic": None}
            acc_start = None
    if current["items"]:
        grouped.append({
            "segment_id": sid,
            "start": current["start"],
            "end": current["end"],
            "text": " ".join(item["text"] for item in current["items"]).strip(),
            "topic": current["topic"] or f"Segment {sid}",
        })
    return grouped


def create_clips(video_path: str | Path, segments: list[dict], clips_dir: str, audio_source: str | Path | None = None) -> list[Path]:
    video_path = Path(video_path).resolve()
    ensure_dir(clips_dir)
    outputs = []
    def _lp(p: Path) -> str:
        s = str(p)
        if os.name == 'nt' and not s.startswith("\\\\?\\"):
            return "\\\\?\\" + s
        return s

    # Probe input for audio presence
    probe = run_cmd(["ffmpeg", "-hide_banner", "-i", _lp(video_path)])
    has_audio = "Audio:" in (probe.stderr or "")
    if not has_audio and audio_source:
        audio_source = Path(audio_source).resolve()
    if not has_audio:
        print(f"⚠️ Warning: input video has no audio stream: {video_path}")

    for s in segments:
        out = Path(clips_dir) / f"clip_{s['segment_id']}.mp4"
        start = float(s["start"])
        end = float(s["end"])
        duration = max(0, end - start)

        if not has_audio and audio_source and audio_source.exists():
            # Trim video and external audio to the same segment and mux
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", _lp(video_path),
                "-ss", str(start), "-i", _lp(Path(audio_source)),
                "-t", str(duration),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                _lp(out)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", _lp(video_path),
                "-t", str(duration),
                "-map", "0",
                "-c:v", "copy",
                "-c:a", "copy",
                _lp(out)
            ]
        res = run_cmd(cmd)
        if res.returncode != 0:
            print(f"ffmpeg stderr:\n{res.stderr}")
            raise RuntimeError(f"ffmpeg clip failed for segment {s['segment_id']}")

        # Verify output contains audio
        vprobe = run_cmd(["ffmpeg", "-hide_banner", "-i", _lp(out)])
        if "Audio:" not in (vprobe.stderr or ""):
            print(f"⚠️ Output clip {out} has no audio stream")
        outputs.append(out)
    return outputs


def run_full_pipeline(url: str | None = None, input_video: str | None = None):
    root = Path(__file__).parent
    cfg_path = root / "config.yaml"
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    downloads = ensure_dir(str(root / paths["downloads"]))
    audio_dir = ensure_dir(str(root / paths["audio"]))
    vocals_dir = ensure_dir(str(root / paths["vocals"]))
    clips_dir = ensure_dir(str(root / paths["clips"]))
    transcripts_dir = ensure_dir(str(root / paths["transcripts"]))

    # Download if URL provided
    video_path = None
    if input_video:
        # sanitize provided path string
        input_video = re.sub(r"\s+\.(\w+)$", r".\1", input_video.strip())
        video_path = Path(input_video)
        
    elif url:
        # yt-dlp with merged A/V into MP4
        template = str(Path(downloads) / "%(title)s.%(ext)s")
        res = run_cmd([sys.executable, "-m", "yt_dlp",
                       "-f", "bv*+ba/best",
                       "--merge-output-format", "mp4",
                       "-o", template,
                       url,
                       "--print", "after_move:filepath"]) 
        if res.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {res.stderr}")
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        video_path = Path(lines[-1])
    else:
        raise ValueError("Provide a URL or input video path for pipeline")

    # Resolve absolute and auto-merge if missing
    video_path = Path(re.sub(r"\s+\.(\w+)$", r".\1", str(video_path))).resolve()
    if not video_path.exists():
        # try alternate with normalized spaces
        alt = Path(str(video_path).replace("  ", " "))
        if alt.exists():
            video_path = alt
        else:
            # try to find nearby file in expected folder
            search_dir = video_path.parent if video_path.is_absolute() else root / "output"
            found = _find_nearby_file(search_dir, video_path.stem, exts=["mp4", "mkv", "webm", "m4a", "mp3"])
            if found:
                video_path = Path(found)
            else:
                # final fallback: use the latest file created in output or downloads
                latest = _latest_file(root / "output", ["mp4", "mkv"]) or _latest_file(root / "downloads", ["mp4", "mkv"])
                if latest:
                    video_path = Path(latest)

        # if still missing, attempt auto-merge
        if not video_path.exists():
            merged = try_auto_merge_from_output(root, desired_name=video_path.name)
            if merged:
                video_path = merged
            else:
                raise RuntimeError(f"Input video not found and auto-merge failed: {video_path}")

    # Steps
    audio_path = extract_audio(video_path, audio_dir)
    try:
        vocals_path, _music_path = separate_vocals(audio_path, vocals_dir)
    except Exception as e:
        print(f"Warning: Vocal separation failed ({e}). Proceeding with original audio.")
        vocals_path = audio_path
    tr_cfg = cfg["transcription"]
    transcript = transcribe_audio(vocals_path, tr_cfg.get("model_size", "small"), tr_cfg.get("compute_type", "int8"))
    transcript_path = Path(transcripts_dir) / f"{video_path.stem}_transcript.json"
    # save with human-friendly timestamps
    formatted = [
        {"id": item.get("id"),
         "start": _fmt_ts(float(item.get("start", 0.0))),
         "end": _fmt_ts(float(item.get("end", 0.0))),
         "text": item.get("text", "").strip()}
        for item in transcript
    ]
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    s_cfg = cfg["silence_removal"]
    clean_audio = remove_silence(vocals_path, audio_dir, s_cfg["stop_periods"], s_cfg["stop_threshold"], s_cfg["detection"]) 
    seg_cfg = cfg["segmentation"]
    segments = segment_transcript(transcript, seg_cfg.get("max_segment_seconds", 60))
    segments_path = Path(transcripts_dir) / f"{video_path.stem}_segments.json"
    # save with human-friendly timestamps
    segments_fmt = []
    for seg in segments:
        seg_fmt = dict(seg)
        if isinstance(seg_fmt.get("start"), (int, float)):
            seg_fmt["start"] = _fmt_ts(float(seg_fmt["start"]))
        if isinstance(seg_fmt.get("end"), (int, float)):
            seg_fmt["end"] = _fmt_ts(float(seg_fmt["end"]))
        segments_fmt.append(seg_fmt)
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_fmt, f, ensure_ascii=False, indent=2)
    clip_paths = create_clips(video_path, segments, clips_dir, clean_audio)
    print("Outputs:")
    print(f"  Transcript: {transcript_path}")
    print(f"  Segments:   {segments_path}")
    print(f"  Clean audio: {clean_audio}")
    for cp in clip_paths:
        print(f"  Clip:       {cp}")


if __name__ == "__main__":
    url = input("Enter YouTube URL (or leave empty to use local file): ").strip()
    local_file = None
    if not url:
        local_file = input("Enter local video path: ").strip()

    # Check if the URL contains a playlist
    is_playlist = "list=" in url if url else False
    if url and is_playlist:
        choice = input("This is a playlist. Do you want to download the entire playlist? (yes/no): ").strip().lower()
        if choice != "yes":
            url = extract_video_url(url)
            is_playlist = False

    if url:
        file_type = input("Enter file type (mp3/mp4): ").strip().lower()
        if file_type not in ["mp3", "mp4"]:
            print("Invalid file type! Defaulting to mp4.")
            file_type = "mp4"

        print("Choose quality: low, medium, high")
        quality = input("Enter quality (default: medium): ").strip().lower() or "medium"

        downloaded_path = download_media(url, file_type, quality, is_playlist)
    else:
        downloaded_path = local_file

    run_full = input("Run full processing pipeline (download→audio→vocals→transcribe→silence→segment→clips)? (yes/no): ").strip().lower()
    if run_full == "yes":
        if url and downloaded_path:
            run_full_pipeline(input_video=downloaded_path)
        elif local_file:
            run_full_pipeline(input_video=local_file)
        else:
            # Fallback: try pipeline with URL directly
            run_full_pipeline(url=url)
    else:
        print("Done with download/conversion only.")