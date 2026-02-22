# AI Context & Progress Log

> **Purpose**: This file maintains the continuity of the development context. Please read this first when starting a new session.
> **Last Updated**: 2026-02-21 21:00 CST

## 🚀 Project Status: Local Deployment Ready & Stable
We have successfully solidified the application for local use. The entire pipeline (Upload -> Slice -> Train -> Generate) is functional, stable, and user-friendly. We have addressed critical dependency issues and provided one-click launcher scripts.

### ✅ Key Achievements (Recent Session)
1.  **One-Click Launcher**: Created `start_app.bat` that:
    *   Auto-detects and activates `.venv`.
    *   Auto-installs missing dependencies (e.g., `torchvision`).
    *   Launches the server in the background and opens the browser automatically.
    *   Displays real-time server logs in the terminal.
2.  **One-Click Shutdown**: Created `stop_app.bat` to cleanly kill the process listening on port 8000.
3.  **Dependency Hell Solved**:
    *   Fixed `ImportError: cannot import name 'cached_download'` by upgrading `diffusers` to `0.36.0`.
    *   Ensured `torchvision` is installed via the startup script logic.
4.  **UI Overhaul (Step 3)**:
    *   Refactored Training UI to a high-contrast "Hacker/Dark" theme.
    *   Fixed Status Sync: "TRAINING" state now displays correctly (no longer stuck on "INITIALIZING").
    *   Fixed Log Display: Real-time logs are visible and readable.
5.  **Data Integrity**:
    *   Implemented **Strict Style Bundling**: Saved styles now permanently include their reference images.
    *   Fixed "No style reference images found" error by ensuring generation pulls strictly from the style folder, not the volatile `processed` directory.

### 🚧 Current Issues & Backlog

#### 1. Immediate UI/Logic Tweaks (Post-Migration)
*   **Variation Slider**: User suspects the slider isn't producing visible changes. Needs code verification in `main.py` / `src/pipeline`.
*   **Step 3 Navigation**: After clicking "View Logs" on the completion card, there is no way to get back to the "Save Style" dialog without refreshing/re-training.

#### 2. Core Quality (Technical Debt)
*   **Background & Grid Artifacts**: Generated characters still sometimes have dark backgrounds or grid lines.
    *   *Solution*: Implement **Adaptive Thresholding** in `utils_slice.py` to purify input data.
*   **Alignment & Punctuation**: Punctuation is vertically centered (wrong for Chinese).
    *   *Solution*: Rewrite `utils.ttf2im` to use font metrics (baseline/descent) instead of simple centering.

## 📋 Next Session Plan (Migration & Polish)

**Priority 1: GitHub Migration**
*   Initialize Git repository.
*   Create `.gitignore` (exclude `.venv`, `outputs/`, `data_examples/upload`, etc.).
*   Commit code and push to remote.

**Priority 2: Minor Logic Fixes**
*   Investigate and fix the Variation Slider logic.
*   Add a "Back to Save" button in the Step 3 Log view.

**Priority 3: Quality Optimization (Future)**
*   Implement Adaptive Thresholding for Step 2.

## 🛠 Command Reference
*   **Start App**: Double-click `start_app.bat`
*   **Stop App**: Double-click `stop_app.bat`
*   **Manual Start**: `.venv\Scripts\python.exe web_ui/main.py`
