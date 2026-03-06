import os
import sys
import json
from urllib.parse import quote
from urllib.request import urlopen
from pathlib import Path
from PIL import Image, ImageDraw


def fetch_hanzi_json(char):
    # 1. Try makemeahanzi local library
    try:
        import makemeahanzi
        graphics = makemeahanzi._load_graphics_file()
        if graphics and char in graphics:
            return graphics[char]
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: makemeahanzi load failed: {e}")

    # 2. Try local cache
    p = Path("web_ui/static/hanzi_data") / f"{char}.json"
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))

    # 3. Fallback to remote CDN (legacy hanzi-writer-data)
    enc = quote(char, safe="")
    url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{enc}.json"
    try:
        with urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data
    except Exception:
        return None


def save_local(char, data):
    d = Path("web_ui/static/hanzi_data")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{char}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return str(path)


def extract_medians(data):
    if 'medians' in data and isinstance(data['medians'], list):
        strokes = []
        for s in data['medians']:
            pts = []
            if isinstance(s, list):
                for p in s:
                    if isinstance(p, dict) and 'x' in p and 'y' in p:
                        pts.append((float(p['x']), float(p['y'])))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
            if pts:
                strokes.append(pts)
        if strokes:
            return strokes
            
    # Fallback to strokes (outlines) ONLY if no medians found
    # This is kept for compatibility but should be avoided
    if 'strokes' in data and isinstance(data['strokes'], list):
        import re
        numre = re.compile(r"-?\d+\.?\d*")
        strokes = []
        for path in data['strokes']:
            nums = numre.findall(path)
            pts = []
            pairs = [float(n) for n in nums]
            for i in range(0, len(pairs)-1, 2):
                pts.append((pairs[i], pairs[i+1]))
            if pts:
                strokes.append(pts)
        if strokes:
            return strokes

    return None


def render_strokes(strokes, out_path, canvas_size=256, pen=6):
    # Collect all points
    allx = [x for s in strokes for (x,y) in s]
    ally = [y for s in strokes for (x,y) in s]
    if not allx:
        return False
    minx, maxx = min(allx), max(allx)
    miny, maxy = min(ally), max(ally)
    w = maxx - minx if maxx>minx else 1.0
    h = maxy - miny if maxy>miny else 1.0
    margin = 0.08
    tgt_w = canvas_size * (1 - 2*margin)
    tgt_h = canvas_size * (1 - 2*margin)
    scale = min(tgt_w / w, tgt_h / h)

    def transform(pt):
        x,y = pt
        nx = (x - minx) * scale + canvas_size*margin
        ny = (y - miny) * scale + canvas_size*margin
        return (nx, ny)

    img = Image.new('RGB', (canvas_size, canvas_size), (255,255,255))
    draw = ImageDraw.Draw(img)
    for s in strokes:
        if len(s) == 1:
            xy = transform(s[0])
            r = max(1, pen//2)
            draw.ellipse([xy[0]-r, xy[1]-r, xy[0]+r, xy[1]+r], fill=(0,0,0))
        else:
            pts = [transform(p) for p in s]
            draw.line(pts, fill=(0,0,0), width=pen)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--char', type=str, default='龙')
    parser.add_argument('--out', type=str, default='outputs/synth_samples/龙_skeleton.png')
    parser.add_argument('--mode', type=str, choices=['rotate','flip'], default='flip')
    args = parser.parse_args()

    ch = args.char
    data = fetch_hanzi_json(ch)
    if data is None:
        print('Failed to fetch hanzi data for', ch)
        return 2
    save_local(ch, data)
    strokes = extract_medians(data)
    if not strokes:
        print('No medians/strokes parsed for', ch)
        return 3
    ok = render_strokes(strokes, args.out, canvas_size=512, pen=8)
    if not ok:
        print('Render failed')
        return 4
    # If requested, flip vertically the saved image (default behavior)
    if args.mode == 'flip':
        try:
            im = Image.open(args.out)
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im.save(args.out)
        except Exception:
            pass
    print('Saved', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
