
import cv2
import numpy as np
import os

def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def cv2_imread(file_path):
    stream = open(file_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

def cv2_imwrite(file_path, img):
    ret, buf = cv2.imencode(".jpg", img)
    if ret:
        with open(file_path, "wb") as f:
            f.write(buf)

def extract_grid_image(image_path, debug=False):
    # Load image
    image = cv2_imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    h, w = image.shape[:2]
    
    # --- Attempt ArUco Detection First ---
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
        
        if ids is not None and len(ids) >= 4:
            # Flatten ids
            ids = ids.flatten()
            
            # Check for required IDs: 0(TL), 1(TR), 2(BR), 3(BL)
            required_ids = [0, 1, 2, 3]
            if all(rid in ids for rid in required_ids):
                print("ArUco Markers Detected. Using ArUco for alignment.")
                
                # Get centers
                centers = {}
                for i, marker_id in enumerate(ids):
                    if marker_id in required_ids:
                        # corners[i] is shape (1, 4, 2)
                        c = corners[i][0]
                        center = np.mean(c, axis=0)
                        centers[marker_id] = center
                        
                src_pts = np.array([
                    centers[0], # TL
                    centers[1], # TR
                    centers[2], # BR
                    centers[3]  # BL
                ], dtype="float32")
                
                # Define Canonical Positions (mm) relative to Grid Top-Left (0,0)
                # Grid Size: 158mm x 78mm
                # Marker Size: 8mm
                # Marker Gap: 4mm
                # Offset = Gap + Size/2 = 4 + 4 = 8mm
                
                # TL Marker (ID 0): 
                # Left of Grid (x = -8mm center)
                # Vertical: Aligned with Prompt line (5mm above grid top).
                # Center: (-8, -5) (Note: Image Y is down, so -5 is up)
                
                # TR Marker (ID 1):
                # Right of Grid (x = 158 + 8 = 166mm center)
                # Vertical: -5mm
                # Center: (166, -5)
                
                # BR Marker (ID 2):
                # Right of Grid (x = 166mm)
                # Vertical: Top aligned with Grid Bottom (78mm).
                # Marker Top = 78mm. Marker Center = 78 + 4 = 82mm.
                # Center: (166, 82)
                
                # BL Marker (ID 3):
                # Left of Grid (x = -8mm)
                # Vertical: 82mm
                # Center: (-8, 82)
                
                # Target Image Scale (Pixels per mm)
                # Let's aim for ~2000px width for the grid (158mm)
                # Scale ~ 12.6 px/mm
                scale = 2000.0 / 158.0 
                
                dst_pts = np.array([
                    [-8, -5],
                    [166, -5],
                    [166, 82],
                    [-8, 82]
                ], dtype="float32") * scale
                
                # Compute Homography
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Output Size: Grid Only (158mm x 78mm)
                out_w = int(158 * scale)
                out_h = int(78 * scale)
                
                warped = cv2.warpPerspective(image, M, (out_w, out_h))
                return warped
                
    except Exception as e:
        print(f"ArUco detection failed (falling back to grid detection): {e}")
    # -------------------------------------

    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000.0 / max(h, w)
        image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        image_resized = image.copy()
        
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Robust Grid Detection Strategy
    # 1. Blur to remove paper texture (Reduced kernel to preserve thin dashed lines)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Adaptive Threshold
    # Reduced C (10 -> 2) to capture lighter/fainter lines. 
    # This makes it more sensitive to noise but ensures dashed lines are detected.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 4)
    
    # Clear borders (20px)
    h_r, w_r = image_resized.shape[:2]
    cv2.rectangle(thresh, (0, 0), (w_r, h_r), (0, 0, 0), 20)
    
    # 3. Morph Close to connect grid lines into a solid blob
    # The new template uses DASHED lines (1pt on, 3pt off).
    # Gap inside cell is ~12mm.
    # We need to connect the dashes into a solid line, AND connect the box into a grid.
    # A small kernel (3x3) is enough to connect dashes.
    # A large kernel (~100px) is needed to connect the grid mesh.
    # Let's stick with the large kernel approach which is robust for both.
    k_size = int(max(h, w) * 0.05) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if not contours:
        raise ValueError("No contours found")
        
    largest = contours[0]
    
    # 5. Get 4 Corners
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        # Fallback to MinAreaRect
        rect = cv2.minAreaRect(largest)
        pts = cv2.boxPoints(rect)
    
    # Scale back
    pts = pts / scale
    pts = np.array(pts, dtype="float32")
    
    warped = four_point_transform(image, pts)
    return warped

def slice_grid(warped_image, output_dir, rows=4, cols=8):
    # Check orientation
    h, w = warped_image.shape[:2]
    
    # If Height > Width, assume rotated 90 degrees.
    # We rotate CLOCKWISE 90 degrees to make it Landscape.
    # Note: This assumption works if the user took the photo with Top-Right or Top-Left orientation.
    # If Top is at Left (common for landscape document in portrait photo), we need CW rotation.
    if h > w:
        print("Detected Vertical Grid. Rotating 90 degrees Clockwise.")
        warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
        # Update h, w
        h, w = warped_image.shape[:2]
        
    # Overwrite the grid_warped.jpg with the rotated version if rotation happened
    # This ensures frontend sees the rotated version
    grid_path = os.path.join(output_dir, "grid_warped.jpg")
    if len(warped_image.shape) == 2:
        warped_color = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
    else:
        warped_color = warped_image
    cv2_imwrite(grid_path, warped_color)
        
    # Grid parameters from PDF generation:
    # Grid w = 18mm, Gap = 2mm
    # Total width = 8*18 + 7*2 = 144 + 14 = 158mm
    # Total height = 4*18 + 3*2 = 72 + 6 = 78mm
    # The detected blob IS the grid (approx).
    # Since we used the grid blob itself, we don't need to crop margin (unless the blob includes markers?)
    # The blob is formed by merging grid lines.
    # So the blob boundary is roughly the outer edge of the grid lines.
    # Let's treat the warped image as the EXACT grid area (158x78).
    
    # Calculate px per mm
    px_per_mm_w = w / 158.0
    px_per_mm_h = h / 78.0
    
    grid_img = warped_image # No margin crop needed if we detected the grid itself
    
    gh, gw = grid_img.shape[:2]
    
    # Characters list (same as PDF)
    chars = [
        '永', '十', '一', '乙', '水', '木',
        '国', '回', '幽', '巫', '用', '月',
        '飞', '也', '戈', '身', '走', '这',
        '神', '韵', '龙', '舞', '灵', '繁', '墨', '魂', '德', '和', '美', '气'
    ]
    
    sliced_paths = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= len(chars):
                continue
                
            char = chars[idx]
            
            # Precise math based on ratios:
            # Cell physical w = 18, gap = 2.
            # Start of cell j: x_mm = j * (18 + 2)
            
            # Using updated pmm
            pmm = gw / 158.0
            pmm_h = gh / 78.0
            
            c_x = int((j * 20) * pmm)
            c_y = int((i * 20) * pmm_h)
            
            c_w = int(18 * pmm)
            c_h = int(18 * pmm_h)
            
            # Boundary checks
            c_x = max(0, min(c_x, gw - 1))
            c_y = max(0, min(c_y, gh - 1))
            c_w = min(c_w, gw - c_x)
            c_h = min(c_h, gh - c_y)
            
            roi = grid_img[c_y:c_y+c_h, c_x:c_x+c_w]
            
            # --- 1. Padding Crop (Asymmetric) ---
            # User request: Top 10% (to avoid guide char), others 5%
            h_roi, w_roi = roi.shape[:2]
            
            pad_top = int(h_roi * 0.10)
            pad_bottom = int(h_roi * 0.05)
            pad_left = int(w_roi * 0.05)
            pad_right = int(w_roi * 0.05)
            
            # Ensure we don't crop to empty
            if h_roi > (pad_top + pad_bottom) and w_roi > (pad_left + pad_right):
                roi = roi[pad_top:h_roi-pad_bottom, pad_left:w_roi-pad_right]
            # --------------------------------------------------
            
            # --- 2. Adaptive Thresholding for Clean Black/White ---
            # Convert to grayscale if not already
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
                
            # Apply Adaptive Threshold to remove light grey grid lines
            # Block size 31, C=15 (Higher C filters more light grey noise)
            # THRESH_BINARY because input is White Paper (High), Ink (Low).
            # Result: Paper -> 255 (White), Ink -> 0 (Black).
            binary_roi = cv2.adaptiveThreshold(
                gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 15
            )
            
            # --- 3. Mask out Guide Character (Top-Left) ---
            # Guide char is at x=2mm, y=4mm(from top) in original 18mm cell.
            # We cropped 5% (~0.9mm) from edges.
            # Remaining guide char is at x~1.1mm, y~1-3mm in cropped cell.
            # We mask the top-left 6mm x 6mm region of the CROPPED cell.
            # This covers the guide char while preserving the center writing area.
            
            mask_mm = 6.0
            # Recalculate pmm for this ROI (approx) or use global pmm
            # We have pmm and pmm_h from outer loop
            mask_px_w = int(mask_mm * pmm)
            mask_px_h = int(mask_mm * pmm_h)
            
            # Safety check: don't mask more than 40% of the cell
            mask_px_w = min(mask_px_w, int(w_roi * 0.4))
            mask_px_h = min(mask_px_h, int(h_roi * 0.4))
            
            if mask_px_w > 0 and mask_px_h > 0:
                binary_roi[0:mask_px_h, 0:mask_px_w] = 255
            # ----------------------------------------------
            
            # Convert back to BGR for consistency with pipeline
            roi = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
            # ---------------------------------------------------
            
            # Save
            out_path = os.path.join(output_dir, f"{idx}_{char}.jpg")
            cv2_imwrite(out_path, roi)
            sliced_paths.append({
                "char": char, 
                "path": out_path, 
                "confidence": 100,
                "bbox": [c_x, c_y, c_w, c_h]
            })
            
    return sliced_paths

def process_scanned_template(image_path, output_dir):
    try:
        # 1. Extract Grid (Robust)
        warped = extract_grid_image(image_path)
        
        # Save warped grid for frontend context
        # Convert to BGR if grayscale for consistency
        if len(warped.shape) == 2:
            warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        else:
            warped_color = warped
            
        grid_path = os.path.join(output_dir, "grid_warped.jpg")
        cv2_imwrite(grid_path, warped_color)
        
        # 2. Slice
        results = slice_grid(warped, output_dir)
        
        return results
    except Exception as e:
        print(f"Error processing template: {e}")
        return []

def crop_and_save(image_path, crop_data, output_dir):
    # crop_data: {char_index, x, y, w, h}
    # image_path: path to grid_warped.jpg
    
    try:
        img = cv2_imread(image_path)
        if img is None:
            return None
            
        x, y, w, h = crop_data['x'], crop_data['y'], crop_data['w'], crop_data['h']
        
        # Validate bounds
        H, W = img.shape[:2]
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = min(w, W - x)
        h = min(h, H - y)
        
        if w <= 0 or h <= 0:
            return None
            
        roi = img[y:y+h, x:x+w]
        
        # Identify filename based on index
        # We need to find the char for this index.
        # This is a bit hacky, better if passed.
        # Assuming filename format: {index}_{char}.jpg
        
        target_filename = None
        char_char = None
        
        files = os.listdir(output_dir)
        prefix = f"{crop_data['index']}_"
        for f in files:
            if f.startswith(prefix) and f.endswith(".jpg"):
                target_filename = f
                char_char = f.split('_')[1].split('.')[0]
                break
        
        if not target_filename:
            return None
            
        out_path = os.path.join(output_dir, target_filename)
        cv2_imwrite(out_path, roi)
        
        return {
            "char": char_char,
            "path": out_path,
            "bbox": [x, y, w, h]
        }
        
    except Exception as e:
        print(f"Crop error: {e}")
        return None
