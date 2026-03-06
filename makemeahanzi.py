import os
import json
from typing import List, Dict, Any

_GRAPHICS_INDEX = None

def _load_graphics_file():
    global _GRAPHICS_INDEX
    if _GRAPHICS_INDEX is not None:
        return _GRAPHICS_INDEX

    paths = [
        os.path.join("third_party", "makemeahanzi", "graphics.txt"),
        os.path.join("third_party", "makemeahanzi", "data", "graphics.txt"),
    ]

    for p in paths:
        if os.path.exists(p):
            idx = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ch = obj.get("character")
                        if ch:
                            idx[ch] = obj
                    except Exception:
                        continue
            _GRAPHICS_INDEX = idx
            return _GRAPHICS_INDEX

    # Not found
    _GRAPHICS_INDEX = {}
    return _GRAPHICS_INDEX


def run_flow(chars: List[str], options: Dict[str, Any] = None):
    """Simple run_flow adapter: returns data items for each char using graphics.txt.
    Each item contains: char, strokes, medians, available, passed, score, retries
    """
    options = options or {}
    graphics = _load_graphics_file()
    items = []
    for ch in chars:
        entry = graphics.get(ch)
        if not entry:
            items.append({
                "char": ch,
                "available": False,
                "strokes": [],
                "medians": [],
                "passed": False,
                "score": 0.0,
                "retries": 0,
            })
            continue

        strokes = entry.get("strokes") or []
        medians = entry.get("medians") or []

        # Basic placeholder scoring: longer medians -> higher score
        avg_len = 0.0
        count = 0
        for m in medians:
            if not m or len(m) < 2:
                continue
            # m is list of [x,y] pairs
            total = 0.0
            prev = m[0]
            for pt in m[1:]:
                dx = pt[0] - prev[0]
                dy = pt[1] - prev[1]
                total += (dx*dx + dy*dy) ** 0.5
                prev = pt
            avg_len += total
            count += 1
        avg_len = (avg_len / max(1, count)) if count else 0.0
        score = min(1.0, (avg_len / 300.0))  # heuristic

        items.append({
            "char": ch,
            "available": True,
            "strokes": strokes,
            "medians": medians,
            "passed": score >= float(options.get("threshold", 0.72)),
            "score": float(score),
            "retries": 0,
        })

    return items
