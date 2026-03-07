import sys
import math
import numpy as np
from synthesize_stroke_sequence import StrokeSynthesizer

def classify_stroke(points):
    """
    Classify stroke type based on geometry of the median points.
    Simple heuristic rule-based classifier.
    Types: Heng (Horizontal), Shu (Vertical), Pie (Left-falling), Na (Right-falling), Dian (Dot), Ti (Rising)
    """
    if not points or len(points) < 2:
        return "Unknown"
        
    pts = np.array(points)
    
    # 1. Vector analysis
    start = pts[0]
    end = pts[-1]
    vec = end - start
    length = np.linalg.norm(vec)
    
    if length < 50: # Threshold depends on canvas size (usually 1024 for hanzi-writer)
        return "Dian (Dot)"
        
    angle_rad = math.atan2(vec[1], vec[0]) # y is down in image, but up in hanzi-writer usually?
    # Let's assume input points are already normalized to image coords (y down)
    # 0 is Right, 90 is Down, 180 is Left, -90 is Up.
    angle_deg = math.degrees(angle_rad)
    
    # 2. Curvature analysis
    # Simple bounding box aspect ratio?
    min_x, max_x = pts[:,0].min(), pts[:,0].max()
    min_y, max_y = pts[:,1].min(), pts[:,1].max()
    w = max_x - min_x
    h = max_y - min_y
    ratio = w / (h + 1e-6)
    
    # 3. Rules
    # Heng: Horizontal-ish (-30 to 30)
    if -30 <= angle_deg <= 30:
        if ratio > 2:
            return "Heng (Horizontal)"
        return "Heng (Short)"
        
    # Ti: Rising (up-right) -> (-90 to -30)
    if -80 <= angle_deg < -30:
        return "Ti (Rising)"
        
    # Shu: Vertical-ish (60 to 120)
    if 60 <= angle_deg <= 120:
        # Check for hooks (Gou) at the end?
        # Look at last segment vs main vector
        if len(pts) > 5:
            last_vec = pts[-1] - pts[-3]
            last_angle = math.degrees(math.atan2(last_vec[1], last_vec[0]))
            diff = abs(last_angle - angle_deg)
            if diff > 90:
                return "ShuGou (Vertical Hook)"
        return "Shu (Vertical)"
        
    # Pie: Left-falling (90 to 180, usually 110-160)
    if 110 <= angle_deg <= 170:
        return "Pie (Left-falling)"
        
    # Na: Right-falling (0 to 90, usually 30-80)
    if 30 < angle_deg < 80:
        return "Na (Right-falling)"
        
    return f"Complex/Other ({angle_deg:.1f}°)"

def main():
    char = '龙'
    if len(sys.argv) > 1:
        char = sys.argv[1]
        
    print(f"Analyzing structure of: {char}")
    
    synth = StrokeSynthesizer()
    data = synth.load_hanzi(char)
    if not data:
        print("Data not found")
        return
        
    raw_strokes = synth.extract_medians(data)
    
    # Normalize to image coords (y down) just for angle consistency
    # synth.normalize_stroke does y-flip.
    bounds = synth.get_char_bounds(raw_strokes)
    norm_strokes = [synth.normalize_stroke(s, bounds) for s in raw_strokes]
    
    print(f"Found {len(norm_strokes)} strokes.")
    print("-" * 30)
    
    for i, s in enumerate(norm_strokes):
        stype = classify_stroke(s)
        # Calculate length for display
        l = np.linalg.norm(np.array(s[-1]) - np.array(s[0]))
        print(f"Stroke {i+1}: {stype:<20} | Len: {l:.1f}")
        
if __name__ == '__main__':
    main()
