import json
import numpy as np
from pathlib import Path
from synthesize_stroke_sequence import StrokeSynthesizer

def build_library():
    print("Building Component Library from Calibration Characters...")
    
    # 1. Load Calibration Map
    try:
        with open('web_ui/static/stroke_map_calibration.json', 'r', encoding='utf-8') as f:
            calib_map = json.load(f)
    except Exception as e:
        print(f"Error loading map: {e}")
        return

    # 2. Initialize Library Structure
    # Format: { "H": [ { "char": "天", "index": 0, "points": [[x,y]...] }, ... ], ... }
    library = {}
    
    synth = StrokeSynthesizer()
    
    # 3. Process Each Character
    total_strokes = 0
    for char, types in calib_map.items():
        data = synth.load_hanzi(char)
        if not data:
            print(f"Warning: No data for {char}")
            continue
            
        raw_strokes = synth.extract_medians(data)
        if not raw_strokes: continue
        
        # Normalize CHAR-level first to keep relative scale?
        # No, for a component library, we usually want STROKE-level normalization (0-1 box)
        # so we can resize it to fit any target skeleton.
        # However, keeping aspect ratio is crucial.
        
        # Strategy: Normalize stroke to fit in 1x1 box, but preserving aspect ratio.
        # Center at (0,0).
        
        for i, s_type in enumerate(types):
            if i >= len(raw_strokes): break
            
            # Get stroke points
            pts = np.array(raw_strokes[i])
            if len(pts) < 2: continue
            
            # Normalize
            # 1. Shift to center
            center = pts.mean(axis=0)
            pts_centered = pts - center
            
            # 2. Scale max dimension to 1.0
            max_dim = np.max(np.abs(pts_centered))
            if max_dim > 0:
                pts_norm = pts_centered / max_dim
            else:
                pts_norm = pts_centered
                
            # Add to library
            if s_type not in library:
                library[s_type] = []
                
            library[s_type].append({
                "char": char,
                "index": i,
                "points": pts_norm.tolist()
            })
            total_strokes += 1
            
    # 4. Save Library
    out_path = Path("data/component_library.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
        
    print(f"Library built. Saved {total_strokes} components to {out_path}")
    print("Component Counts:")
    for k, v in library.items():
        print(f"  {k}: {len(v)}")

if __name__ == '__main__':
    build_library()
