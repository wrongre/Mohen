import os
import re
import json
from pathlib import Path
from PIL import Image, ImageDraw


def load_hanzi(char):
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
    
    # 3. Fallback fetch
    from urllib.parse import quote
    from urllib.request import urlopen
    enc = quote(char, safe='')
    url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{enc}.json"
    try:
        with urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            # Cache it
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            return data
    except Exception:
        return {}


def parse_stroke_path(path_str):
    # extract all numbers and group into pairs
    nums = re.findall(r"-?\d+\.?\d*", path_str)
    vals = [float(n) for n in nums]
    pts = []
    for i in range(0, len(vals)-1, 2):
        pts.append((vals[i], vals[i+1]))
    return pts


def normalize_strokes(strokes, canvas_size=512, margin=0.08):
    # strokes: list of list of (x,y)
    allx = [x for s in strokes for (x,y) in s]
    ally = [y for s in strokes for (x,y) in s]
    if not allx:
        return strokes
    minx, maxx = min(allx), max(allx)
    miny, maxy = min(ally), max(ally)
    w = maxx - minx if maxx>minx else 1.0
    h = maxy - miny if maxy>miny else 1.0
    tgt_w = canvas_size * (1 - 2*margin)
    tgt_h = canvas_size * (1 - 2*margin)
    scale = min(tgt_w / w, tgt_h / h)

    def transform(pt):
        x,y = pt
        nx = (x - minx) * scale + canvas_size*margin
        ny = (y - miny) * scale + canvas_size*margin
        return (nx, ny)

    return [[transform(p) for p in s] for s in strokes]


def render_cumulative(strokes_norm, out_dir, canvas_size=512, pen=12):
    os.makedirs(out_dir, exist_ok=True)
    imgs = []
    n = len(strokes_norm)
    for i in range(1, n+1):
        img = Image.new('RGB', (canvas_size, canvas_size), (255,255,255))
        draw = ImageDraw.Draw(img)
        for j in range(i):
            s = strokes_norm[j]
            if len(s) == 1:
                x,y = s[0]
                r = max(1, pen//2)
                draw.ellipse([x-r,y-r,x+r,y+r], fill=(0,0,0))
            else:
                # draw thicker line and round caps by drawing circles at points
                draw.line(s, fill=(0,0,0), width=pen)
                r = max(1, pen//2)
                for (x,y) in s:
                    draw.ellipse([x-r,y-r,x+r,y+r], fill=(0,0,0))
        fname = f"step_{i}.png"
        fpath = os.path.join(out_dir, fname)
        img.save(fpath)
        imgs.append(fpath)
    return imgs


def concat_horizontal(image_paths, out_path):
    imgs = [Image.open(p).convert('RGB') for p in image_paths]
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h), (255,255,255))
    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_im.save(out_path)
    return out_path


def main(char='龙'):
    data = load_hanzi(char)
    raw_strokes = []
    
    # Priority: medians > strokes
    if 'medians' in data and isinstance(data['medians'], list) and len(data['medians']) > 0:
        for s in data['medians']:
            pts = []
            for p in s:
                # Handle [x, y] list format
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
                # Handle {"x":..., "y":...} dict format
                elif isinstance(p, dict) and 'x' in p:
                    pts.append((float(p['x']), float(p['y'])))
            if pts:
                raw_strokes.append(pts)
    
    # Fallback to strokes ONLY if no medians
    if not raw_strokes and 'strokes' in data and isinstance(data['strokes'], list):
        print(f"Warning: No medians found for {char}, falling back to stroke outlines.")
        for s in data['strokes']:
            if isinstance(s, str):
                pts = parse_stroke_path(s)
                if pts:
                    raw_strokes.append(pts)
            elif isinstance(s, list):
                # maybe array of points
                pts = []
                for p in s:
                    if isinstance(p, dict) and 'x' in p:
                        pts.append((float(p['x']), float(p['y'])))
                    elif isinstance(p, (list, tuple)) and len(p)>=2:
                        pts.append((float(p[0]), float(p[1])))
                if pts:
                    raw_strokes.append(pts)
                    
    if not raw_strokes:
        print(f"Error: No stroke data found for {char}")
        return None, []

    # Default: vertically flip strokes coordinates so output matches flipped skeleton expectation
    strokes_norm = normalize_strokes(raw_strokes, canvas_size=512, margin=0.08)
    # apply vertical flip to normalized strokes (flip Y relative to canvas)
    flipped = []
    for s in strokes_norm:
        flipped.append([(x, 512 - y) for (x,y) in s])
    strokes_norm = flipped
    out_dir = os.path.join('outputs', 'synth_samples', f'{char}_stroke_steps')
    imgs = render_cumulative(strokes_norm, out_dir, canvas_size=512, pen=14)
    concat = os.path.join('outputs', 'synth_samples', f'{char}_strokes_concat.png')
    concat_horizontal(imgs, concat)
    print('Saved', concat)
    for p in imgs:
        print('-', p)
    return concat, imgs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--char', type=str, default='龙')
    args = parser.parse_args()
    main(args.char)
