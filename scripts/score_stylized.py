import numpy as np
import math
from PIL import Image

def compute_iou(mask_a, mask_b):
    # mask is numpy array (0 or 255/1)
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0: return 0.0
    return float(inter) / float(union)

def evaluate_stroke(ref_pts, cand_pts, params):
    """
    Score a candidate stroke against a reference stroke (standard skeleton).
    Returns dict of scores.
    """
    scores = {}
    
    # 1. Position Score (Center Distance + Start Point Distance)
    # User feedback: Start point is critical for calligraphy structure.
    ref_arr = np.array(ref_pts)
    cand_arr = np.array(cand_pts)
    
    if len(ref_arr) < 2 or len(cand_arr) < 2:
        return {'total': 0, 'reason': 'Too short'}
        
    # A. Centroid Distance
    ref_center = ref_arr.mean(axis=0)
    cand_center = cand_arr.mean(axis=0)
    dist_center = np.linalg.norm(ref_center - cand_center)
    
    # B. Start Point Distance
    ref_start = ref_arr[0]
    cand_start = cand_arr[0]
    dist_start = np.linalg.norm(ref_start - cand_start)
    
    # Combined Score (Max 30)
    # Weight Start Point more heavily (e.g. 50/50 split of the 30 points)
    # Tolerance: Center 25px, Start 25px
    score_center = max(0, 15 * (1 - dist_center / 25.0))
    score_start = max(0, 15 * (1 - dist_start / 25.0))
    
    pos_score = score_center + score_start
    scores['pos'] = pos_score
    
    # 2. Direction Score (Angle)
    # Only for strokes with sufficient length
    ref_vec = ref_arr[-1] - ref_arr[0]
    cand_vec = cand_arr[-1] - cand_arr[0]
    ref_len = np.linalg.norm(ref_vec)
    cand_len = np.linalg.norm(cand_vec)
    
    if ref_len < 20: # Short stroke (dot)
        # Angle doesn't matter much for dots
        dir_score = 20.0
    else:
        # Cosine similarity
        cos_sim = np.dot(ref_vec, cand_vec) / (ref_len * cand_len + 1e-6)
        # angle difference in degrees
        angle_diff = math.degrees(math.acos(min(1.0, max(-1.0, cos_sim))))
        # Tighter check: 20 degrees max
        # Score: 20 * (1 - angle/20)
        dir_score = max(0, 20 * (1 - angle_diff / 25.0))
        
    scores['dir'] = dir_score
    
    # 3. Shape Score (Chamfer / Euclidean on resampled)
    # Simple check: length ratio
    len_ratio = cand_len / (ref_len + 1e-6)
    # Ideal ratio 1.0. Allow 0.7 ~ 1.3
    dist_from_1 = abs(len_ratio - 1.0)
    # Score: 20 * (1 - dist/0.4)
    shape_score = max(0, 20 * (1 - dist_from_1 / 0.5))
    scores['shape'] = shape_score
    
    # 4. Correctness (Topology / General Direction)
    # Check if the general flow direction matches the reference.
    # Crucial for Pie (Left-falling) vs Na (Right-falling).
    # If direction is reversed or deviated > 90 degrees, this is topologically wrong.
    
    # Use vectors from start to end
    # (Already computed as ref_vec, cand_vec)
    
    if ref_len < 10 or cand_len < 10:
        # Too short to determine flow direction reliably
        corr_score = 30.0
    else:
        # Cosine similarity of the main flow vector
        flow_cos = np.dot(ref_vec, cand_vec) / (ref_len * cand_len + 1e-6)
        
        if flow_cos < 0:
            # Direction is reversed or > 90 deg deviation!
            # Severe penalty. This is likely a mirrored stroke or wrong type.
            corr_score = 0.0
        elif flow_cos < 0.5:
            # Deviation between 60 and 90 degrees
            # Linear penalty from 30 down to 0
            # cos=0.5 -> 60deg -> score 30? No, let's say 60deg is barely acceptable.
            # Let's map cos [0, 1] to score [0, 30]
            corr_score = 30.0 * (flow_cos ** 2) # Quadratic falloff?
            # Or simple linear: 30 * flow_cos
            corr_score = 30.0 * flow_cos
        else:
            # Deviation < 60 degrees. Acceptable topology.
            corr_score = 30.0

    scores['corr'] = corr_score
    
    # Total
    total = pos_score + dir_score + shape_score + corr_score
    scores['total'] = min(100.0, total)
    
    return scores

def check_fail_gates(scores, weights):
    # weights: {'pos': 18, 'dir': 12, 'shape': 0, 'corr': 20}
    fails = []
    if scores['pos'] < weights['pos']: fails.append('pos')
    if scores['dir'] < weights['dir']: fails.append('dir')
    if scores['corr'] < weights['corr']: fails.append('corr')
    # shape ignored
    return fails
