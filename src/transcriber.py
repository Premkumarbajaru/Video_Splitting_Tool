from pathlib import Path
from typing import List, Dict


def transcribe(audio_path: str | Path, model_size: str = "small", compute_type: str = "int8") -> List[Dict]:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("faster-whisper not installed. Please install requirements.txt") from e

    audio_path = str(audio_path)
    model = WhisperModel(model_size, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, vad_filter=True)

    result = []
    for i, seg in enumerate(segments):
        result.append({
            "id": i + 1,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })
    return result
