from typing import List, Dict


def segment_transcript(segments: List[Dict], max_segment_seconds: int = 60) -> List[Dict]:
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
