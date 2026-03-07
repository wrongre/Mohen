## Short-term Prototype Plan (Record): Stroke-Level Stylization Prototype — 2026-03-06 (Update)

Objective: Rapidly implement a stroke-by-stroke generation prototype to validate the "Generate -> Score -> Retry -> Lock" control flow, using a lightweight `style-profile` instead of a full model initially.

Core Concept:
- Use `makemeahanzi` skeleton as the structural baseline.
- Extract `style-profile` (slant, width, curvature) from user samples (e.g., Yan Style).
- Synthesize stroke variants geometrically (Affine transforms + Noise).
- Score variants based on Position, Direction, Shape, and Correctness.
- Optimize via a retry loop until a high-quality stroke is found.

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

**Next Steps**:
- Sync code to GitHub.
- Prepare for integration with the main Web UI.
- (Future) Replace geometric synthesizer with Diffusion-based renderer using the optimized skeleton as ControlNet input.

---
## 📝 Session Log — 2026-03-05
... (Previous content remains)
