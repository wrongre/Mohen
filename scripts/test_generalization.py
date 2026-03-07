import json
import numpy as np
from synthesize_stroke_sequence import StrokeSynthesizer
from PIL import Image, ImageDraw, ImageFont
import math
from classify_stroke_types import COLOR_MAP

def load_component_library():
    try:
        with open('data/component_library.json', 'r', encoding='utf-8') as f:
            library = json.load(f)
        return library
    except Exception as e:
        print(f"Error loading library: {e}")
        return None

def normalize_stroke(points):
    """
    Normalize a single stroke to a canonical 1x1 box (centered),
    consistent with how the library was built.
    """
    pts = np.array(points)
    if len(pts) < 2: return pts
    
    # 1. Shift to center
    center = pts.mean(axis=0)
    pts_centered = pts - center
    
    # 2. Scale max dimension to 1.0
    max_dim = np.max(np.abs(pts_centered))
    if max_dim > 0:
        pts_norm = pts_centered / max_dim
    else:
        pts_norm = pts_centered
        
    return pts_norm

def classify_stroke_nearest_neighbor(stroke_points, library):
    """
    Classify a stroke by finding the nearest neighbor in the library.
    Metric: Mean Squared Error (MSE) between resampled points?
    Or Chamfer Distance?
    
    Since strokes have different point counts, we should resample them to a fixed count (e.g., 20 points).
    """
    target_norm = normalize_stroke(stroke_points)
    target_resampled = resample_stroke(target_norm, 32) # Resample to 32 points
    
    best_type = "Unknown"
    min_dist = float('inf')
    best_match_info = ""
    
    for s_type, examples in library.items():
        for ex in examples:
            lib_pts = np.array(ex['points'])
            # Resample library stroke too (though we could cache this)
            lib_resampled = resample_stroke(lib_pts, 32)
            
            # Calculate distance (MSE)
            # Simple Euclidean distance between corresponding points
            dist = np.mean(np.sum((target_resampled - lib_resampled)**2, axis=1))
            
            if dist < min_dist:
                min_dist = dist
                best_type = s_type
                best_match_info = f"{ex['char']}[{ex['index']}]"
                
    return best_type, best_match_info, min_dist

def resample_stroke(points, num_points=32):
    """
    Resample a stroke to a fixed number of points using linear interpolation.
    """
    if len(points) < 2:
        return np.resize(points, (num_points, 2))
        
    # Calculate cumulative distance along the stroke
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len == 0:
        return np.resize(points, (num_points, 2))
        
    # Generate target distances
    target_dists = np.linspace(0, total_len, num_points)
    
    # Interpolate
    new_points = np.zeros((num_points, 2))
    new_points[:, 0] = np.interp(target_dists, cum_dists, points[:, 0])
    new_points[:, 1] = np.interp(target_dists, cum_dists, points[:, 1])
    
    return new_points

def main():
    test_chars = "天道酬勤"
    print(f"Testing generalization on: {test_chars}")
    
    library = load_component_library()
    if not library: return
    
    synth = StrokeSynthesizer()
    
    # Visualization setup
    cols = len(test_chars)
    cell_size = 512 # Higher res
    img = Image.new('RGB', (cols * cell_size, cell_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        big_font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = None
        big_font = None

    for idx, char in enumerate(test_chars):
        print(f"Processing {char}...")
        x_offset = idx * cell_size
        
        # Load raw data
        data = synth.load_hanzi(char)
        if not data:
            print(f"  No data for {char}")
            continue
            
        raw_strokes = synth.extract_medians(data)
        if not raw_strokes: continue
        
        # Draw bounding box
        draw.rectangle([x_offset, 0, x_offset + cell_size - 1, cell_size - 1], outline=(200, 200, 200))
        
        # Normalize for display
        bounds = synth.get_char_bounds(raw_strokes)
        display_strokes = [synth.normalize_stroke(s, bounds) for s in raw_strokes]
        scale = cell_size / 512.0
        
        classifications = []
        
        for i, stroke in enumerate(raw_strokes):
            # CLASSIFY using Nearest Neighbor against Library
            s_type, match_info, dist = classify_stroke_nearest_neighbor(stroke, library)
            classifications.append(s_type)
            
            print(f"  Stroke {i+1}: {s_type} (Match: {match_info}, Dist: {dist:.4f})")
            
            # Draw stroke
            d_stroke = display_strokes[i]
            pts = [(p[0] * scale + x_offset, p[1] * scale) for p in d_stroke]
            
            # Map shorthand to full name for color map
            mapping = {
                'H': 'Heng', 'S': 'Shu', 'P': 'Pie', 'N': 'Na', 
                'D': 'Dian', 'T': 'Ti', 'SW': 'ShuWanGou', 
                'HZ': 'HengZhe', 'HP': 'HengPie', 'SZ': 'ShuZhe',
                'W': 'WoGou'
            }
            full_type = mapping.get(s_type, 'Unknown')
            base_type = full_type # simplified
            color = COLOR_MAP.get(base_type, (100, 100, 100))
            
            if len(pts) > 1:
                draw.line(pts, fill=color, width=4)
                # Start point
                sx, sy = pts[0]
                draw.ellipse([sx-4, sy-4, sx+4, sy+4], fill=color)
                
                # Label index and type
                label = f"{i+1}:{s_type}"
                draw.text((sx+10, sy), label, fill=(0, 0, 0), font=font)
        
        # Draw char title
        draw.text((x_offset + 10, 10), char, fill=(0, 0, 0), font=big_font)
        
    out_path = "outputs/generalization_test_天道酬勤.png"
    img.save(out_path)
    print(f"Saved test result to {out_path}")

if __name__ == "__main__":
    main()
