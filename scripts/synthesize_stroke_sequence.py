import os
import re
import json
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

try:
    import makemeahanzi
except ImportError:
    makemeahanzi = None


class StrokeSynthesizer:
    def __init__(self, profile_path='web_ui/static/style_profiles/MyStyle.json'):
        self.profile = self._load_profile(profile_path)
        self.canvas_size = 512
        self.margin = 0.08

    def _load_profile(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('aggregate', {})
        except Exception:
            return {}

    def get_style_params(self):
        agg = self.profile
        # Base params from profile
        pen_mean = agg.get('pen_width', {}).get('mean', 3.0)
        pen_std = agg.get('pen_width', {}).get('std', 0.8)
        slant_mean = agg.get('slant_deg', {}).get('mean', 0.0)
        slant_std = agg.get('slant_deg', {}).get('std', 5.0)
        curv_mean = agg.get('curvature_mean_deg', {}).get('mean', 20.0)

        # Sample for this stroke
        pen_w = max(1.5, random.gauss(pen_mean, pen_std))
        slant = random.gauss(slant_mean, slant_std)
        
        # Derived params
        # Scale: thinner strokes -> slightly larger scaling to fill space? 
        # Actually usually consistent scaling. Let's vary slightly around 1.0
        scale_x = random.gauss(1.0, 0.05)
        scale_y = random.gauss(1.0, 0.05)
        
        return {
            'pen_width': pen_w,
            'slant_deg': slant,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'offset_x': random.gauss(0, 5.0), # small random shift
            'offset_y': random.gauss(0, 5.0),
            'noise_sigma': max(0.5, pen_std * 0.5)
        }

    def load_hanzi(self, char):
        # 1. Try makemeahanzi local library
        if makemeahanzi:
            try:
                graphics = makemeahanzi._load_graphics_file()
                if graphics and char in graphics:
                    return graphics[char]
            except Exception:
                pass

        # 2. Try local cache
        p = Path("web_ui/static/hanzi_data") / f"{char}.json"
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
            
        # 3. Remote fallback
        from urllib.parse import quote
        from urllib.request import urlopen
        enc = quote(char, safe='')
        url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{enc}.json"
        try:
            with urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                return data
        except Exception:
            return {}

    def extract_medians(self, data):
        raw_strokes = []
        if 'medians' in data and isinstance(data['medians'], list):
            for s in data['medians']:
                pts = []
                for p in s:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
                if pts:
                    raw_strokes.append(pts)
        return raw_strokes

    def normalize_stroke(self, stroke, bounds):
        # normalize a single stroke using pre-calculated bounds
        minx, maxx, miny, maxy = bounds
        w = maxx - minx if maxx > minx else 1.0
        h = maxy - miny if maxy > miny else 1.0
        
        tgt_w = self.canvas_size * (1 - 2*self.margin)
        tgt_h = self.canvas_size * (1 - 2*self.margin)
        scale = min(tgt_w / w, tgt_h / h)
        
        # Center in canvas
        off_x = (self.canvas_size - w*scale) / 2 - minx*scale
        off_y = (self.canvas_size - h*scale) / 2 - miny*scale
        # Actually usually we map [minx, miny] to [margin, margin] relative to top-left?
        # Let's align to standard canvas center for now
        
        # Standard Hanzi Writer data is 1024x1024 typically, with y up?
        # We need to flip y for image coords usually (0,0 at top-left)
        # But let's keep normalization consistent first.
        
        # Re-implement normalization relative to bounding box of the WHOLE CHAR
        # to preserve relative stroke positions
        
        new_pts = []
        # Hanzi Writer / Makemeahanzi data is usually in a coordinate system where Y goes UP (mathematical)
        # or at least needs flipping to match standard image coordinates (Y down).
        # We also need to center it.
        
        # Calculate offset to center the character bounding box
        content_w = (maxx - minx) * scale
        content_h = (maxy - miny) * scale
        
        offset_x = (self.canvas_size - content_w) / 2
        offset_y = (self.canvas_size - content_h) / 2
        
        for x,y in stroke:
            # X: standard scaling + centering
            nx = (x - minx) * scale + offset_x
            
            # Y: FLIP vertically! (1024 - y) logic relative to bounds
            # Map [miny, maxy] to [0, content_h] but FLIPPED
            # So maxy -> 0 (top), miny -> content_h (bottom)
            # Formula: (maxy - y) * scale
            ny = (maxy - y) * scale + offset_y
            
            new_pts.append((nx, ny))
        return new_pts

    def apply_transform(self, stroke, params):
        if not stroke: return []
        pts = np.array(stroke)
        
        # Center of this stroke (for local rotation/scale)
        center = pts.mean(axis=0)
        rel = pts - center
        
        # 1. Scale
        sx, sy = params.get('scale_x', 1.0), params.get('scale_y', 1.0)
        rel = rel * np.array([sx, sy])
        
        # 2. Slant (Shear/Rotation)
        # Slant usually means horizontal shear, or slight rotation
        deg = params.get('slant_deg', 0.0)
        rad = math.radians(deg)
        # Shear matrix for slant: x' = x + y * tan(theta)
        # But rotation is safer for now
        c, s = math.cos(rad), math.sin(rad)
        rot = np.array([[c, -s], [s, c]])
        rel = rel @ rot.T
        
        # 3. Offset
        ox = params.get('offset_x', 0.0)
        oy = params.get('offset_y', 0.0)
        
        # 4. Noise
        sigma = params.get('noise_sigma', 0.0)
        if sigma > 0:
            noise = np.random.normal(0, sigma, rel.shape)
            rel += noise
            
        return (rel + center + np.array([ox, oy])).tolist()

    def generate_stroke_image(self, stroke_pts, params, size=None):
        if size is None: size = self.canvas_size
        img = Image.new('L', (size, size), 0) # Black background, White stroke
        draw = ImageDraw.Draw(img)
        
        if not stroke_pts:
            return img
            
        pen_w = params.get('pen_width', 3.0)
        
        # Draw variable width stroke?
        # For now, simple line with rounded caps
        # To simulate pressure: start thin, middle thick, end thin?
        # Or simple constant width for geometric baseline
        
        coords = [(x,y) for x,y in stroke_pts]
        if len(coords) < 2:
            return img
            
        # Draw main spine
        draw.line(coords, fill=255, width=int(pen_w))
        # Draw caps
        r = pen_w / 2
        draw.ellipse([coords[0][0]-r, coords[0][1]-r, coords[0][0]+r, coords[0][1]+r], fill=255)
        draw.ellipse([coords[-1][0]-r, coords[-1][1]-r, coords[-1][0]+r, coords[-1][1]+r], fill=255)
        
        return img

    def get_char_bounds(self, strokes):
        allx = [x for s in strokes for x,y in s]
        ally = [y for s in strokes for x,y in s]
        if not allx: return (0,1024,0,1024)
        return (min(allx), max(allx), min(ally), max(ally))

