import os
import sys
import json
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

def otsu_thresh(img):
    """
    Convert PIL/numpy image to binary (0/255) using Otsu's method.
    Foreground is 255.
    """
    if cv2 is None:
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = arr.astype(np.uint8)
        th = arr.mean() # Fallback
        return (arr < th).astype(np.uint8) * 255
    
    # Convert to grayscale numpy array
    arr = np.array(img.convert('L'))
    # Threshold (assuming dark text on light background, so invert)
    # cv2.THRESH_BINARY_INV means: if src > thresh, dst=0, else 255. 
    # Otsu automatically finds thresh.
    thresh_val, bin_img = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img

def zhang_suen_thinning(binary):
    """
    Skeletonize binary image (foreground=255).
    """
    if cv2 is not None:
        # Use OpenCV's thinning if available (faster and standard)
        # ximgproc is in opencv-contrib-python, might not be installed.
        # Fallback to morphological thinning if ximgproc is missing.
        try:
            return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            pass
            
    # Python implementation of Zhang-Suen
    img = (binary // 255).astype(np.uint8)
    prev = np.zeros(img.shape, np.uint8)
    
    while True:
        # Step 1
        m1 = np.zeros(img.shape, np.uint8)
        # We use correlation to find matches. But explicit loop is clearer for the specific conditions.
        # To optimize, we can use convolution, but for now let's stick to a slightly optimized loop or just keep it simple.
        # Given the previous implementation was slow but working, let's use a standard optimized approach if possible.
        # Actually, for 128x128 images, the python loop is "okay" but slow. 
        # Let's try to use a slightly faster method using padded views.
        
        # Padded view
        padded = np.pad(img, 1, mode='constant')
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]
        
        # Neighbors
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        
        # Transitions 0->1
        # p2->p3, p3->p4, ... p9->p2
        A = ((p2 == 0) & (p3 == 1)).astype(int) + \
            ((p3 == 0) & (p4 == 1)).astype(int) + \
            ((p4 == 0) & (p5 == 1)).astype(int) + \
            ((p5 == 0) & (p6 == 1)).astype(int) + \
            ((p6 == 0) & (p7 == 1)).astype(int) + \
            ((p7 == 0) & (p8 == 1)).astype(int) + \
            ((p8 == 0) & (p9 == 1)).astype(int) + \
            ((p9 == 0) & (p2 == 1)).astype(int)
            
        # Conditions
        c1 = (img == 1)
        c2 = (B >= 2) & (B <= 6)
        c3 = (A == 1)
        c4 = (p2 * p4 * p6 == 0)
        c5 = (p4 * p6 * p8 == 0)
        
        marker = c1 & c2 & c3 & c4 & c5
        img[marker] = 0
        
        # Step 2
        # Recompute neighbors/transitions not strictly needed if we didn't change them yet? 
        # No, we just changed img. Need to recompute based on new img.
        padded = np.pad(img, 1, mode='constant')
        p2, p3, p4, p5 = padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:]
        p6, p7, p8, p9 = padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2]
        
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        A = ((p2 == 0) & (p3 == 1)).astype(int) + \
            ((p3 == 0) & (p4 == 1)).astype(int) + \
            ((p4 == 0) & (p5 == 1)).astype(int) + \
            ((p5 == 0) & (p6 == 1)).astype(int) + \
            ((p6 == 0) & (p7 == 1)).astype(int) + \
            ((p7 == 0) & (p8 == 1)).astype(int) + \
            ((p8 == 0) & (p9 == 1)).astype(int) + \
            ((p9 == 0) & (p2 == 1)).astype(int)
            
        c1 = (img == 1)
        c2 = (B >= 2) & (B <= 6)
        c3 = (A == 1)
        c4_2 = (p2 * p4 * p8 == 0)
        c5_2 = (p2 * p6 * p8 == 0)
        
        marker = c1 & c2 & c3 & c4_2 & c5_2
        img[marker] = 0
        
        if not marker.any():
            break
            
    return img * 255

def build_skeleton_graph(ske):
    """
    Convert skeleton image to a set of paths (list of points).
    Returns: list of lists of (y, x) coordinates.
    """
    # 1. Identify junctions and endpoints
    # 3x3 convolution to count neighbors
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    
    # Normalize skeleton to 0/1
    ske_bool = (ske > 127).astype(np.uint8)
    
    if cv2 is not None:
        neighbors = cv2.filter2D(ske_bool, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    else:
        # simple manual convolution fallback if needed, but we assume numpy
        from scipy.signal import convolve2d
        neighbors = convolve2d(ske_bool, kernel, mode='same', boundary='fill', fillvalue=0)
    
    # Mask by skeleton to only care about skeleton points
    neighbors = neighbors * ske_bool
    
    # Endpoints: 1 neighbor
    # Junctions: > 2 neighbors
    # Regular path points: 2 neighbors
    
    # We want to segment the skeleton into "strokes" (segments between junctions/endpoints)
    # Strategy: Remove junctions, then find connected components (segments), then reconnect?
    # Or simple traversal.
    
    # Let's try "Remove Junctions" approach which is robust for graph extraction
    junctions = (neighbors > 2)
    # Also treat tight clusters of junctions as one? For now simple.
    
    # Subtract junctions from skeleton
    segments_map = ske_bool.copy()
    segments_map[junctions] = 0
    
    # Label connected components (these are the strokes)
    from scipy.ndimage import label
    labeled_segments, num_features = label(segments_map, structure=np.ones((3,3)))
    
    paths = []
    
    for i in range(1, num_features + 1):
        # Get coordinates of this segment
        coords = np.argwhere(labeled_segments == i)
        if len(coords) < 3: # Ignore tiny specks
            continue
            
        # Order the points in the segment to form a line
        # Simple heuristic: find an endpoint (neighbor count in segment == 1) and walk
        # Within the segment (which has no junctions), every point has <= 2 neighbors.
        # Endpoints of the segment have 1 neighbor (within the segment).
        
        # Local neighbor count within segment
        seg_mask = (labeled_segments == i).astype(np.uint8)
        if cv2 is not None:
            seg_neigh = cv2.filter2D(seg_mask, -1, kernel, borderType=cv2.BORDER_CONSTANT) * seg_mask
        else:
            seg_neigh = convolve2d(seg_mask, kernel, mode='same', boundary='fill', fillvalue=0) * seg_mask
            
        seg_ends = np.argwhere(seg_neigh == 1)
        
        if len(seg_ends) == 0:
            # Closed loop (circle)
            start = coords[0]
        else:
            start = seg_ends[0]
            
        # Walk
        path = [tuple(start)]
        current = tuple(start)
        visited = set([current])
        
        while len(path) < len(coords):
            # Look for neighbor in coords that is not visited
            # 8-connectivity
            found = False
            r, c = current
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    if (nr, nc) not in visited and seg_mask[nr, nc]:
                        current = (nr, nc)
                        path.append(current)
                        visited.add(current)
                        found = True
                        break
                if found: break
            if not found:
                break
                
        paths.append(path)
        
    return paths

def analyze_paths(paths, dist_map):
    """
    Calculate metrics from paths.
    """
    slants = []
    curvatures = []
    widths = []
    
    for path in paths:
        if len(path) < 5:
            continue
            
        pts = np.array(path)
        
        # --- Width ---
        # Sample width along path
        # dist_map values are distance to background. Pen width approx 2 * dist.
        path_widths = [dist_map[p[0], p[1]] * 2.0 for p in path]
        widths.extend(path_widths)
        
        # --- Slant ---
        # Fit a line to the path or take start-end vector
        # Only consider "vertical-ish" strokes for slant calculation
        # Vector from start to end
        dy = pts[-1][0] - pts[0][0]
        dx = pts[-1][1] - pts[0][1]
        length = math.hypot(dx, dy)
        
        if length < 10: # Ignore short strokes
            continue
            
        angle_rad = math.atan2(dy, dx) # y is down, x is right. 
        # Vertical is dy large, dx small. 
        # angle of vertical line (going down) is pi/2 (90 deg).
        # We want deviation from vertical.
        
        angle_deg = math.degrees(angle_rad)
        
        # Map to -90 to 90 range relative to x-axis
        # But we want deviation from vertical axis (Y-axis).
        # Vertical down = 90 deg. 
        # Deviation = angle - 90.
        # If line goes up (-90), deviation is also relative to vertical.
        
        # Let's normalize vector to point "down" (dy > 0)
        if dy < 0:
            dy = -dy
            dx = -dx
            
        # Now vector is in lower half plane.
        # Angle is 0 to 180. Vertical is 90.
        angle_deg = math.degrees(math.atan2(dy, dx))
        slant_deviation = 90 - angle_deg # Positive = leans left (like /), Negative = leans right (like \) ? 
        # Wait, usually slant is defined: Positive = leans right (Italic).
        # If vector is (1, 2) -> x=1, y=2. Leans right. 
        # atan2(2, 1) = 63 deg. 90 - 63 = 27 deg.
        # So (90 - angle) gives positive for right-lean. Correct.
        
        # Filter: Only consider strokes that are roughly vertical to calculate "Script Slant"
        # e.g., within +/- 30 degrees of vertical (tightened from 45 to exclude diagonals like Pie/Na).
        if abs(slant_deviation) < 30:
            # Weight by length
            slants.append((slant_deviation, length))
            
        # --- Curvature ---
        # Sum of absolute angle changes
        # Smooth path first?
        # Simple finite difference
        if len(pts) > 4:
            # Subsample or smooth to reduce quantization noise
            # Take vectors every k steps
            step = max(1, len(pts) // 10)
            vectors = []
            for i in range(0, len(pts) - step, step):
                v_dy = pts[i+step][0] - pts[i][0]
                v_dx = pts[i+step][1] - pts[i][1]
                vn = math.hypot(v_dx, v_dy)
                if vn > 1e-6:
                    vectors.append((v_dx/vn, v_dy/vn))
            
            total_angle = 0
            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i+1]
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)
                total_angle += angle
            
            # Curvature metric: radians per pixel length? Or total turning angle?
            # "Curvature Mean" usually refers to average curvature.
            # Let's use total_angle_degrees / path_length_pixels * 100 (deg per 100 px)
            if length > 0:
                curv_metric = math.degrees(total_angle) / length * 10.0 # deg per 10 pixels
                curvatures.append((curv_metric, length))

    return slants, curvatures, widths

def analyze_image(path, resize_to=128):
    try:
        img = Image.open(path).convert('L')
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return None
        
    # Resize keeping aspect ratio, pad to square
    # But for analysis, distortion might not matter much if we just want stroke properties.
    # However, stretching changes slant. So we must preserve aspect ratio.
    
    # Resize such that smaller dim is resize_to? Or fit within?
    # Let's fit within resize_to x resize_to
    img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    
    # Paste into square canvas to simplify processing
    new_img = Image.new('L', (resize_to, resize_to), (255)) # White bg
    # Center
    paste_x = (resize_to - img.width) // 2
    paste_y = (resize_to - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    binary = otsu_thresh(new_img)
    
    # Distance transform (on binary, fg=255)
    # cv2.distanceTransform calculates distance to nearest ZERO pixel.
    # So we want distance from FG to BG.
    # binary has FG=255.
    if cv2 is not None:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    else:
        # Fallback distance transform
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(binary > 127)
        
    ske = zhang_suen_thinning(binary)
    
    paths = build_skeleton_graph(ske)
    slants_weighted, curvs_weighted, widths = analyze_paths(paths, dist)
    
    result = {
        'path': str(path),
        'pen_widths': widths,
        'slants': slants_weighted, # list of (angle, weight)
        'curvatures': curvs_weighted
    }
    return result

def weighted_stats(pairs):
    if not pairs:
        return None
    values = np.array([p[0] for p in pairs])
    weights = np.array([p[1] for p in pairs])
    
    mean = np.average(values, weights=weights)
    var = np.average((values - mean)**2, weights=weights)
    std = math.sqrt(var)
    
    return {'mean': float(mean), 'std': float(std)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_dir', type=str, default='data_examples/train/TargetImage/MyStyle')
    parser.add_argument('--out', type=str, default='data_examples/style_profiles/MyStyle.json')
    parser.add_argument('--resize', type=int, default=128)
    args = parser.parse_args()
    
    style_path = Path(args.style_dir)
    if not style_path.exists():
        print(f"Style dir not found: {style_path}")
        return
        
    images = [p for p in style_path.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
    if not images:
        print("No images found")
        return
        
    all_widths = []
    all_slants = []
    all_curvs = []
    
    image_results = []
    
    print(f"Processing {len(images)} images...")
    for p in images:
        res = analyze_image(p, args.resize)
        if res:
            all_widths.extend(res['pen_widths'])
            all_slants.extend(res['slants'])
            all_curvs.extend(res['curvatures'])
            
            # Lightweight summary for per-image entry
            img_summary = {
                'path': res['path'],
                'slant_mean': weighted_stats(res['slants'])['mean'] if res['slants'] else 0,
                'width_mean': float(np.mean(res['pen_widths'])) if res['pen_widths'] else 0
            }
            image_results.append(img_summary)
            print(f".", end='', flush=True)
            
    print("\nCalculating aggregates...")
    
    # Aggregate stats
    profile = {
        'meta': {'style_dir': str(style_path), 'n_images': len(image_results)},
        'images': image_results,
        'aggregate': {}
    }
    
    if all_widths:
        w_arr = np.array(all_widths)
        profile['aggregate']['pen_width'] = {
            'mean': float(np.mean(w_arr)),
            'std': float(np.std(w_arr)),
            'p50': float(np.median(w_arr))
        }
    
    if all_slants:
        s_stats = weighted_stats(all_slants)
        profile['aggregate']['slant_deg'] = s_stats
        
    if all_curvs:
        c_stats = weighted_stats(all_curvs)
        profile['aggregate']['curvature_mean_deg'] = c_stats
        
    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
        
    print(f"Profile saved to {args.out}")
    print("Aggregate Results:")
    print(json.dumps(profile['aggregate'], indent=2))

if __name__ == '__main__':
    main()
