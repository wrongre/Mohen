## Short-term Prototype Plan (Record): Stroke-Level Stylization Prototype — 2026-03-07 (Update)

Objective: Rapidly implement a stroke-by-stroke generation prototype to validate the "Generate -> Score -> Retry -> Lock" control flow, using a lightweight `style-profile` instead of a full model initially.

Core Concept:
- Use `makemeahanzi` skeleton as the structural baseline.
- Extract `style-profile` (slant, width, curvature) from user samples (e.g., Yan Style).
- Synthesize stroke variants geometrically (Affine transforms + Noise).
- Score variants based on Position, Direction, Shape, and Correctness.
- Optimize via a retry loop until a high-quality stroke is found.

### 📅 Session Log — 2026-03-07

**Status**:
- **Critical Milestone**: Established a "Component Library" (Component-Based Stylization) using 30 verified "Golden Set" characters (MoHen Calibration).
    - **Problem**: Makemeahanzi skeleton data often diverges from standard stroke taxonomy (e.g., "Horizontal" vs "Pie").
    - **Solution**: Manually calibrated 30 representative characters (天道酬勤 + 26 core chars) to build a "Ground Truth" mapping.
    - **Result**: Extracted ~216 normalized stroke components (H, S, P, N, etc.) into `data/component_library.json`.
- **Generalization Verified**:
    - Created `scripts/test_generalization.py` to classify unseen characters ("天道酬勤") using Nearest Neighbor matching against the Component Library.
    - Achieved high accuracy in identifying complex strokes (e.g., "Dao" bottom, "Chou" side) without any character-specific rules.
    - **Insight**: The 30-char calibration set is sufficient to cover most stroke types, acting as a "Rosetta Stone" for the dataset.
- **Stroke Taxonomy Refinement**:
    - Identified "Ti" (Rise) strokes in the dataset are often geometrically flat (indistinguishable from Horizontal).
    - **Strategy**: Will use context-aware classification (or manually add "Hai" to the library) to better distinguish "Ti".
- **Next Steps**:
    - **Style Extraction**: Extract style profiles (curvature, thickness, etc.) from user uploads.
    - **Stylization**: Warp the standard components from the library to fit the user's style and the target skeleton.

### 📅 Session Log — 2026-03-06

**Status**:
- **Critical Fix**: Rewrote `scripts/extract_style_profile.py`.
    - **Problem**: Previous PCA-based slant calculation yielded ~69° (invalid) for Yan style.
    - **Fix**: Implemented Skeleton Path Tracing + Vertical Stroke Filtering (±30°).
    - **Result**: New slant is **-2.26°** (near vertical), perfectly matching Yan style.
- **Implementation**:
    - Created `scripts/optimize_char_flow.py`: Implements the "Generate -> Score -> Retry" loop.
    - Created `scripts/score_stylized.py`: Implements scoring logic (Pos, Dir, Shape, Corr).
    - Refactored `scripts/synthesize_stroke_sequence.py`: Geometric synthesis engine.
- **Scoring Improvements**:
    - **Pos**: Now penalizes both Centroid deviation AND Start Point deviation (50/50 weight).
    - **Corr**: Added vector direction check (Cosine Similarity). If angle > 90°, score drops to 0 to prevent reversed strokes (Pie/Na).
- **Verification**:
    - Generated "Dragon" (龙) with the new pipeline.
    - Produced `outputs/compare_dragon.png`: 3-layer overlay (Yan Red + Standard Gray + Generated Black).
    - Confirmed that the optimization loop effectively filters out bad strokes (e.g., wrong direction).

**Known Issues**:
- **Single Stroke Collapse**: Occasionally, one stroke (e.g., Stroke 1 or 2 of "Dragon") fails to reach a high score even after retries, or settles on a suboptimal "best" candidate (e.g., Score ~75 but visually poor).
    - *Hypothesis*: Random sampling range might be too wide/narrow, or the scoring landscape has local maxima.
    - *Future Work*: Consider Simulated Annealing or more intelligent parameter adjustment during retries.

**Key Insights (Style Definition)**:
- **Layout vs. Style**: A font's legibility depends on the general Layout (Skeleton), which is relatively stable across fonts.
- **Style Differentiators**: Style is primarily defined by:
    1.  **Start/End Positions (Pos)**: Critical. e.g., Yan style has short, high starts; skeletons might be lower/longer.
    2.  **Slant**: Overall tilt of the character.
    3.  **Shape/Curvature**: Specific stroke trajectories (e.g., length of the hook in "Vertical Curve Hook").
    4.  **Width**: (For brush styles) Variation in stroke thickness.
- **Implication**: Future optimization should prioritize **Start/End Point accuracy** over general centroid alignment, as this is where "Style" lives.

**Strategic Shift: Shape First (Component Assembly)**:
- **Concept**: Instead of "Layout First -> Shape Adjustment" (perturbing a line), we propose "Shape First -> Fit to Skeleton".
- **Method**:
    1.  **Stroke Classification**: Identify skeleton strokes as Heng, Shu, Pie, Na, etc. (Verified feasible via `scripts/classify_stroke_types.py`).
    2.  **Component Retrieval**: Fetch a prototypical "Yan-style Heng" or "Yan-style Pie" (from a pre-extracted library or clustered user inputs).
    3.  **Warping**: Apply Affine or TPS (Thin Plate Spline) transformation to fit the prototype shape onto the target skeleton's start/end points.
- **Goal**: Achieve strong "Flesh/Texture" stylization (e.g., Yan's thick hooks) even if structural alignment requires some relaxation. This mimics `mxFont`'s component approach but at the stroke level.

**Classification Standard (Taxonomy)**:
- **Reference**: Use standard Chinese "Stroke Input Method" (笔画输入法) categories as the baseline.
- **Complexity**: The component set is small (< 50 types), similar to Hiragana.
- **Granularity**: 
    - e.g., Short Heng and Medium Heng are likely the same component (scaled).
    - Long Heng might be distinct due to specific shape features (e.g., curvature).
- **Dynamics**: Components should not be static; they must support parametric variation to simulate dynamic writing (avoiding "dead/rigid" repetition).

**Systemic Challenge: Skeleton-Taxonomy Discrepancy**:
- **Problem**: Standard stroke databases (Map) and actual skeleton data (MakeMeAHanzi) often disagree. A dictionary may define a stroke as a "Pie" while the skeleton is geometrically a "Horizontal". Manual per-char adjustment is unscalable.
- **Proposed Solution (Hybrid Alignment)**: 
    - Instead of hard lookup, use **Semantic-Geometric Alignment**.
    - 1. Retrieve the standard stroke sequence (e.g., "H, S, P...").
    - 2. Perform geometric feature extraction on all segments of the actual skeleton.
    - 3. Use an alignment algorithm (e.g., Greedy or DTW) to find the best mapping between the "ideal" sequence and the "real" geometry.
    - This allows the system to correctly assign a "Pie" component to a "Horizontal-looking" skeleton segment based on its position in the stroke order.

**Alternative Strategy: Data-Driven Stroke Clustering (Bottom-Up)**:
- **Concept**: Instead of forcing external taxonomy onto the skeleton data, analyze the skeleton dataset itself to find its inherent "primitive shapes".
- **Method**: 
    1. Extract thousands of strokes from `makemeahanzi`.
    2. Normalize and cluster them (e.g., K-Means).
    3. Identify the ~50 distinct clusters that naturally emerge (e.g., "Flat Horizontal", "Vertical Hook", "Short Pie").
- **Benefit**: This creates a taxonomy that is *native* to the skeleton data, eliminating the geometric discrepancy. We then map these native clusters to style components.
- **Mapping to Standard**: While the clustering is data-driven, the final components must be labeled using the standard taxonomy (H, S, P, N) to preserve semantic meaning and compatibility with input methods. The result is a "Data-Driven Geometry mapped to Standard Semantics".

**Applicability to Cursive/Running Scripts**:
- **Hidden Logic**: Even in Cursive (Cao) or Running (Xing) scripts, there is an underlying stroke logic (e.g., connected strokes are just standard strokes with "ligatures").
- **Adaptability**: The "Shape First" approach is valid for these scripts too. The components might be more abstract or connected, but the principle of "Skeleton + Component" remains. The classifier just needs to recognize "connected stroke" types.

**Next Steps**:
- Sync code to GitHub.
- Prepare for integration with the main Web UI.
- (Future) Replace geometric synthesizer with Diffusion-based renderer using the optimized skeleton as ControlNet input.

---
## 📝 Session Log — 2026-03-05
... (Previous content remains)
