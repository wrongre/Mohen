
import numpy as np
import sys

def test_skeleton_boundary_issue():
    print("Testing skeleton boundary issue...")
    
    # Create a 10x10 mask
    H, W = 10, 10
    seg_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Case 1: Bottom edge stroke.
    # Pixels at (9, 2), (9, 3), (9, 4)
    # If we are at (9, 3), neighbor check includes (10, 3) which is OOB.
    
    coords = np.array([[9, 2], [9, 3], [9, 4]])
    for r, c in coords:
        seg_mask[r, c] = 1
        
    start = (9, 2)
    path = [start]
    current = start
    visited = set([start])
    
    # Simulate the loop in extract_style_profile.py
    try:
        while len(path) < len(coords):
            found = False
            r, c = current
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    
                    # Original code: no boundary check
                    # if (nr, nc) not in visited and seg_mask[nr, nc]:
                    
                    # NOTE: Python's behavior:
                    # seg_mask[10, 3] -> IndexError
                    # seg_mask[-1, 3] -> Valid (wraps to last row)
                    
                    # To demonstrate IndexError (Case 1):
                    # When r=9, dr=1 -> nr=10. seg_mask[10, ...] raises IndexError.
                    
                    if (nr, nc) not in visited and seg_mask[nr, nc]:
                        current = (nr, nc)
                        path.append(current)
                        visited.add(current)
                        found = True
                        break
                if found: break
            if not found:
                break
        print("Case 1 (Bottom Edge): Finished without error (Unexpected if bug exists)")
    except IndexError:
        print("Case 1 (Bottom Edge): Caught expected IndexError!")
    except Exception as e:
        print(f"Case 1 (Bottom Edge): Caught unexpected exception: {e}")

    # Case 2: Wrap-around issue (Negative Index)
    # Stroke at top edge (0, 2), (0, 3), (0, 4)
    # And a pixel at bottom edge (9, 3)
    # If we are at (0, 3), dr=-1 -> nr=-1 (index 9). 
    # If seg_mask[9, 3] is 1, it might jump there.
    
    seg_mask_2 = np.zeros((H, W), dtype=np.uint8)
    coords_2 = np.array([[0, 2], [0, 3], [0, 4]])
    for r, c in coords_2:
        seg_mask_2[r, c] = 1
    
    # Add a "ghost" pixel at the bottom that shouldn't be connected
    seg_mask_2[9, 3] = 1 
    
    start = (0, 3) # Start middle
    path = [start]
    current = start
    visited = set([start])
    
    print("\nTesting Case 2 (Negative Index Wrap-around)...")
    jumped = False
    
    while len(path) < 10: # limit iterations
        found = False
        r, c = current
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                
                # Original code
                if (nr, nc) not in visited and seg_mask_2[nr, nc]:
                    if nr < 0 or nc < 0:
                        print(f"  Jumped from {current} to ({nr}, {nc}) which wraps to ({nr%H}, {nc%W})")
                        jumped = True
                    
                    current = (nr, nc)
                    path.append(current)
                    visited.add(current)
                    found = True
                    break
            if found: break
        if not found:
            break
            
    if jumped:
        print("Case 2: Confirmed wrap-around bug!")
    else:
        print("Case 2: Did not wrap around (maybe due to neighbor order)")

if __name__ == "__main__":
    test_skeleton_boundary_issue()
