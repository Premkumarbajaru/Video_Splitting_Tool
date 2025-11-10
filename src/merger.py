from pathlib import Path
from typing import Optional
from .utils import ensure_dir, run_cmd


def _base_key(p: Path) -> str:
    name = p.name
    idx = name.rfind('.f')
    return name[:idx] if idx != -1 else p.stem


def find_and_merge_streams(search_dir: str | Path, processed_dir: str | Path, output_name: Optional[str] = None) -> Optional[Path]:
    """Find separate video-only (.mp4) and audio-only (.webm) streams in search_dir and merge to processed_dir.
    Returns merged file path or None if not found."""
    search_dir = Path(search_dir)
    processed_dir = Path(processed_dir)
    ensure_dir(str(processed_dir))

    videos = list(search_dir.glob('*.mp4'))
    audios = list(search_dir.glob('*.webm'))
    if not videos or not audios:
        return None

    # Group by base key (title before .f###)
    video_map = {}
    for v in videos:
        key = _base_key(v)
        # keep largest file per key
        if key not in video_map or v.stat().st_size > video_map[key].stat().st_size:
            video_map[key] = v
    audio_map = {}
    for a in audios:
        key = _base_key(a)
        if key not in audio_map or a.stat().st_size > audio_map[key].stat().st_size:
            audio_map[key] = a

    # Find a matching pair
    for key, v in video_map.items():
        if key in audio_map:
            a = audio_map[key]
            out_name = output_name or (key + '_merged.mp4')
            # sanitize: replace forbidden chars in Windows filename
            safe = out_name.replace('|', '_')
            out_path = processed_dir / safe
            cmd = [
                'ffmpeg', '-y',
                '-i', str(v),
                '-i', str(a),
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest', str(out_path)
            ]
            res = run_cmd(cmd)
            if res.returncode != 0:
                raise RuntimeError(f'ffmpeg merge failed: {res.stderr}')
            return out_path
    return None
