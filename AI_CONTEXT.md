## 短期原型计划（记录）：逐笔风格化原型  — 2026-03-05

目的：在没有训练好风格模型的前提下，快速落地一个可运行的逐笔书写原型，用以验证“按笔生成→评分→重试→锁定”的控制流和可视化体验，便于后续替换为学习型模块。
核心思路：
- 使用用户约 30 字样本提取轻量 `style-profile`（统计笔宽、起点偏移、倾角分布、曲率/墨迹密度等），作为风格近似。 
- 以 `makemeahanzi` / `hanzi-writer` 的骨架（中线/笔画路径）为目标骨架，对每一笔生成若干参数化变体（起点/角度/尺度/路径扰动/笔触宽度）。
- 对每个候选渲染并评分：综合风格相似度 + 骨架符合度 + 稳定性惩罚；若最高分 ≥ 阈值则接受并锁定， 否则对高分候选做变异再试（最多 N 次）。
- 前端以逐笔 SVG 动画播放写入过程，显示每笔尝试记录与是否被接受，允许交互调整阈值以便调参。

短期实现步骤（优先级与交付物）：
1. 数据与样本处理：上传或采集 ~30 字样本；标准化尺寸并提取简单统计特征（风格 profile）。
2. 候选生成器（服务端）：对骨架每笔生成 K 个变体（平移/旋转/缩放/噪声/笔触参数）。
3. 评分函数（服务端）：实现 raster + 向量混合评分（风格相似度 + 距离到骨架 + 稳定性惩罚）。
4. 流程控制器 API：`POST /api/optimize_char`，返回每字每笔的候选与最终 accepted 路径及尝试记录。 
5. 前端可视化：在 `generate.html` 中实现逐笔 SVG 播放器并绑定 `/api/optimize_char` 输出（每笔动画 + 分数信息 + 阈值调节）。
6. 迭代：用可视化反馈调节评分与变异策略；随后替换为学习型 style-encoder / 条件生成器（中期目标）。

接口草案（最小版）：
- `POST /api/optimize_char` 输入：{ text, threshold?, max_retries?, k_candidates? }
- 返回：{ items: [ { char, strokes: [ { path, stroke_style, score, accepted, attempts:[{score,params}] } ], passed } ] }

短期里程碑（交付验收）：
- 服务端：候选生成 + 单笔评分 + 单字优化流程并提供调试输出。 
- 前端：每字逐笔动画展示，能看到 accepted / retried 状态与分数，阈值可调。

下一步（基于你偏好先看可视化）：我将先实现前端逐笔 SVG 播放器并把现有 `/api/stroke_flow_run` 或新 `/api/optimize_char` 的输出映射到写字格进行播放。你确认我现在开始实现前端可视化吗？
# AI Context & Progress Log

> **Purpose**: This file maintains the continuity of the development context. Please read this first when starting a new session.
> **Last Updated**: 2026-02-28 23:20 CST

## 🚀 Project Status: Bug Fixes & Stability Phase
Critical startup & training bugs fixed. Full training pipeline now working end-to-end. UI remains polished and functional.

### 🌙 End-of-Day Wrap-up (2026-02-27)

Session focus:
- Shifted `Generate` page from "batch output viewer" to "writing-process preview" for upcoming stroke-level pipeline.
- Confirmed product direction: final goal is article-level writing flow, auto stroke scoring, and auto retry/skip logic (no user per-stroke approval).

Completed UI refactor (Generate page):
1. Removed low-value controls for current stage:
   - Removed `Inference Preset` and `Preset Version` from user-facing UI.
   - Removed floating `Real-time Writing Parameters` panel.
2. Moved paper template control to left sidebar and simplified options to two templates:
   - `400 Grid Manuscript`
   - `21-line Letter Paper` (lightweight illustrative mode, non-primary)
3. Rebuilt preview area for writing workflow:
   - Added scaffold-based writing surface with reference characters (衬字).
   - Added status strip (current char / progress / stroke / score / retries / failed count).
   - Added failed-character list for skip-and-analyze loop.
4. 400-grid paper correctness pass:
   - Fixed full-grid continuity and border closure issues.
   - Enforced 20 columns per row.
   - Pre-rendered full 400 slots even before generation.
5. Interaction upgrade:
   - Added zoom+pan usability for enlarged paper.
   - Added hand tool button + space-hold temporary pan.
   - Added pan boundary clamp and reset behavior.

Known visual notes:
- Reference-character size was reduced, but may still need minor per-template tuning.
- `21-line Letter Paper` is intentionally minimal now; detailed realism is deferred.

Next-session priority (2026-02-28):
1. Integrate Make Me a Hanzi as the first stroke-order knowledge source.
2. Build a minimal "see how the character is written" demo path (visual stroke sequence).
3. Start backend contract for stroke-level scoring/threshold pass without changing the current article input UX.

### 🔁 Deferred Mechanism Update (2026-02-28)

During stroke-flow demo iteration, user requested a "discuss later" mechanism for intentionally skipped scope.

Implemented policy in Generate demo:
- Punctuation is now **auto-deferred** in stroke flow (no retry loop, no fail count inflation).
- Deferred items are recorded in a dedicated UI section: **Deferred (Discuss Later)**.
- Failed-character list remains focused on structural errors; deferred list is for planned-later items.

Rationale:
- Keeps main path focused on non-punctuation character correctness (primary pain point: wrong characters).
- Prevents punctuation/no-stroke data from polluting failure analytics.

### ⚙️ Stroke Pipeline Demo Milestone (2026-02-28)

Major integration completed:
- Added Make Me a Hanzi integration endpoint in backend (`/api/stroke_order`) with cache + local-first/remote-fallback behavior.
- Connected Generate UI to live stroke-order data and stroke playback.
- Added stroke-direction hints (start/end markers + direction trend text).

UI/interaction updates during this milestone:
- Generate page layout transitioned to writing-focused flow; removed nested-card visual clutter in status area.
- Added hand-drag canvas navigation (button + space-hold pan) for zoomed paper workflow.
- Added 400-grid and letter-paper templates with scaffold rendering for immediate visual context.

Flow demo status:
- Implemented `RUN STROKE FLOW (DEMO)` end-to-end control path:
   - text -> char -> stroke loop
   - per-stroke scoring -> retry -> pass/skip
   - failed-character and deferred-item collection

Scoring model direction (agreed with user):
- Prioritize "do not write wrong characters" over rigid positional template fitting.
- Use a two-level logic:
   1. Correctness gate (`C`) as mandatory pass condition.
   2. Style soft score (`S`) preserving writer individuality (fat/thin, high/low, slant differences).
- Current implementation exposes `C / S / F` style of evaluation in demo status text.

Training decision at this stage:
- Do NOT retrain immediately before validating stroke-flow control loop.
- Next hard phase: evaluate whether current whole-character training can support stable stroke-level attachment;
   if not, move to stroke-aware supervision / retraining plan.

### 🌙 End-of-Day Wrap-up (2026-02-25)

Session result:
- Multiple inference iterations completed (`v1` / `v2` / `v3`) with measurable quality improvements.
- PNG export clipping issue confirmed fixed.
- Core issue persists but narrowed: simple-character structural stability is still the main bottleneck.

Latest user confirmation:
- With Yan Zhenqing-style regular-script training samples, error rate drops significantly.
- Remaining residual errors are localized (simple characters + punctuation behavior + horizontal drift in some cases).
- Overall improved, but still below final clone-quality target.

Next-session intent (2026-02-26):
- Continue testing with additional calligraphy fonts to map style-vs-accuracy trade-offs.
- Keep focus on core clone quality first; postpone non-critical UI expansion.

### 🧪 Multi-Style Generalization Check (Yan style) - 2026-02-26

`style/v3` user feedback:
- Wrong list includes: `入`->`人`, `江`, `湖` (月 component extra stroke), `王` (horizontal drift), `间` (extra stroke), `生` (horizontal drift).
- No raw/base glyph direct-output issue in this run.
- Horizontal-stroke-to-right-falling drift still present.
- Punctuation is acceptable but can be slightly more stylized.

Applied follow-up tuning (generalized, non-hardcoded):
1. Reduced style-v3 distortion amplitude (noise + angle), increased structural guidance and steps.
2. Strengthened horizontal-stroke protection weights for simple/medium complexity characters.
3. Slightly increased punctuation stylization range in style-v3 while preserving punctuation correctness floor.

### 🧪 Yan Style Follow-up (2026-02-26, latest)

Latest user validation:
1. Horizontal-drift issue in specific characters still persists (bottom horizontal remains hard to recover in some cases).
2. `胜` left component improved, but right-side `生` was negatively affected (trade-off observed).
3. Punctuation stylization is inconsistent: first comma shows style, other punctuation remains close to baseline.

Interim decision:
- Do not over-focus on the stubborn horizontal-drift subset in this stage (may be tied to Yan training sample characteristics).
- Treat this as a style-dependent limitation and continue broader generalization evaluation.

### 🧪 Running Script Evaluation (2026-02-26)

User result summary:
- Character accuracy is still low (only about 6~7 characters correct in this run).
- `人` / `入` confusion did not appear in this sample.
- Horizontal-to-right-falling drift (横->捺) not observed here, suggesting this issue may be style-dependent rather than global.

Style consistency observation:
- First three sentences appear stylistically consistent.
- Fourth sentence differs from the first three, but is visually closest to source handwriting style.

Implication for next tuning:
- Need better line/segment-level style consistency control (avoid sentence-to-sentence style drift).
- Preserve current gains on confusion pairs while improving overall running-script accuracy.

### 🧪 Validation Log (2026-02-25)
**Preset Test: `fidelity` (文字保真)**

User-reported wrong characters in current run:
- `天`
- `志`
- `弗`
- `山`
- `不`
- `也`

Notes:
- Keep this set as a regression baseline for upcoming clone-quality tuning.
- Requirement confirmed: solve issues incrementally without breaking previously fixed behavior.

**Preset Test: `balanced` (平衡)**

User-reported wrong/suspicious characters in current run:
- `天`
- `志`
- `弗`
- `山`
- `海` (uncertain)
- `不`
- `也`

Observations:
- Overall close to `fidelity` result set.
- Deformation is slightly larger than `fidelity`, but still acceptable.

**New Bug Found (pending, do not fix yet):**
- Exported PNG is incomplete / clipped compared with on-screen preview.
- Keep as backlog until `style` preset validation is finished.

**Preset Test: `style` (风格优先)**

User-reported wrong/suspicious characters in current run:
- `天`
- `志`
- `无`
- `远` (close)
- `弗` (closer than in `balanced`)
- `届` (close)
- `海` (close)
- `不`

Cross-preset observations:
- `fidelity` / `balanced` / `style` are currently too similar; differences are not instantly recognizable.
- Error pattern is consistent: simple characters tend to get extra strokes; missing strokes are less common.
- Hypothesis: running-script connected-stroke style information is over-applied to simple structures, causing additive pen traces.

### 🧪 Validation Log Addendum (2026-02-25)
**Version Compare: `v1/fidelity` vs `v2/fidelity`**

User observations:
- Wrong-character set changed between versions.
- `v1/fidelity` currently has fewer wrong characters than `v2/fidelity`.
- Character `道` consistently shows extra strokes near the lower part of the 辶 component.

New issues identified (record only, no fix yet):
1. **Punctuation missing region**: punctuation glyph areas are sometimes not fully generated.
2. **Punctuation-area noise**: significant noise appears around punctuation zones, especially near grid-line boundaries.

Priority note:
- Keep these as dedicated cleanup tasks after current version comparison wrap-up.

### 🧪 Three-Group Comparative Notes (2026-02-25)

User-side visual comparison summary:
- `style` preset starts to show correct direction in some glyphs, but overall style dominance is still not strong enough.
- Positive case in `style`: character `精` looks best; 米字旁 connected strokes are accurate; right-side `青` is also correct.
- Positive case in `style`: final `不` has interesting/closer style behavior.
- Negative case in `style`: both occurrences of `志` are wrong, while the second `志` is correct in both fidelity versions.
- Structural confusion remains: `入` and `人` skeletons are not separated reliably; outputs collapse mostly to `人`.

Action focus derived from this comparison:
1. Keep strengthening style only where it improves connected radicals (e.g., 米字旁), avoid global over-application.
2. Add confusion-pair guardrails for `入` vs `人` and targeted glyph checks for `志`.
3. Preserve fidelity-friendly decoding path for characters where style causes structural collapse.

### 🧪 Cross-Text Regression Check (2026-02-25)

Status update from a new input text set:
- PNG export clipping bug: **fixed/verified**.
- Core character-accuracy issue persists across text sets.

Observed persistent failures:
- `入` still frequently collapses to `人`.
- `一` remains unstable/incorrect.
- `王` can be over-drawn (observed as 5 horizontal strokes).

General pattern (stable across different test text):
- Complex characters: lower error rate (not zero).
- Simple characters: very high error rate (roughly around half in user observation).

Conclusion:
- Current v1/v2/v3 tuning improves behavior only partially; root issue for simple-character structure preservation is still unresolved.

### 🧪 New Style Validation (Yan Zhenqing / Regular Script Bias) - 2026-02-25

Result summary:
- Overall character correctness improved significantly compared with previous running-script-heavy samples.
- Most characters are correct; residual issues are now localized.

Residual issues observed:
1. Punctuation issue after "所趋": expected comma (，) but generated mark contains extra artifact/noise.
2. Character `勤`: extra dot-like stroke appears.
3. Character `无`: structure looks slightly abnormal.

Interpretation:
- Regular-script training samples strongly improve baseline structural correctness.
- Remaining errors are now mostly punctuation cleanup and small additive-stroke artifacts.

### 🧪 v3 Iteration Feedback (2026-02-25, latest)

`v3/fidelity`:
- `一` and `入` are correct now (but almost no stylization).
- `王` / `江` / `业`: horizontal stroke trend drifts into a right-falling stroke (横 -> 捺 tendency).
- `辈` incorrect.
- Most other characters are acceptable.

`v3/style`:
- `雄` / `辈` show extra horizontal stroke.
- `江` incorrect.
- Second occurrence of `一` incorrect.
- Overall acceptable direction; still has strong horizontal-to-right-falling drift.

Punctuation:
- Current punctuation is too close to standard font (overly rigid).
- Requirement update: keep punctuation correctness first, but allow slight stylization instead of full standard-glyph rigidity.

Next micro-tuning targets (no architecture change):
1. Add horizontal-stroke protection to prevent 横 being over-transformed to 捺.
2. Add extra-stroke suppression focus for `辈`/`雄`-like structures (generalized rule, not hard-coded chars).
3. Add tiny style perturbation for punctuation in `v3` (very low amplitude), while preserving comma/period correctness.

### ✅ Evening Update (2026-02-24)
**🎯 Focus: Variation Perceptibility + Progressive Generation UX**

1. **Variation Control Refactor (Completed)**
   * Upgraded variation behavior from weak global shift to stronger perceptual mapping
   * Added character-level deterministic variation and style perturbation for clearer differences
   * Unified variation scale to **0.00 ~ 1.00** end-to-end (UI slider + API + backend mapping)
   * Result: difference is now clearly visible in practical testing

2. **Progressive Generation (Completed)**
   * Generation now runs in chunks and appends results progressively
   * Users no longer wait on a blank canvas for full completion
   * RTX 4060 laptop validation: 40+ characters in ~2-3 minutes is acceptable

3. **A/B Compare Mode for Debugging (Completed)**
   * Added A/B compare for variation tuning (`0.20 vs 0.80`)
   * Confirmed effective for parameter tuning and visual validation
   * Note: keep as **debug capability**, not required in final product UX

4. **Risk Identified: Clone Quality Gap**
   * Current outputs can contain wrong characters and style drift toward regular script
   * Primary concern moved from UI polish to **handwriting cloning fidelity**
   * Hypothesis: not training collapse, likely inference/content-style balance and decoding constraints

### 📌 Priority Decision for Next Session
**Priority should be CLONE QUALITY before UI polish.**

Recommended order:
1. OCR-based wrong-character metrics + sampling validation
2. Inference profile tuning (content/style balance presets)
3. Character consistency/re-sampling strategy
4. Then resume UI enhancement tasks

### ✅ Key Achievements (Current Session - 2026-02-24)
**🐛 CRITICAL BUG FIXES**

1. **Startup Connection Error - ROOT CAUSE FIXED**:
   * **Problem**: First connection attempt always fails, then succeeds after a few seconds
   * **Root Cause**: `start_app.bat` waited only 5 seconds before opening browser, but FastAPI/uvicorn needed more time
   * **Solution Implemented**:
     - Replaced blind `timeout 5` with intelligent connection detection
     - Added retry loop checking if server is actually responding (max 60 retries)
     - Added `-UseBasicParsing` to PowerShell to avoid security prompts
     - Result: Fully automated startup with **ZERO user interaction**

2. **Training Pipeline Failure (Step 3) - FIXED**:
   * **Problem**: Training crashed immediately at step 3 with `NotADirectoryError`
   * **Root Cause**: `dataset/font_dataset.py` line 40 - code assumed all items in `TargetImage/` were directories, but `.gitkeep` file (for preserving empty dirs) was treated as a directory
   * **Solution**: 
     - Added `os.path.isdir()` check before attempting to list directory contents
     - Skip non-directory items (files like `.gitkeep`)
     - File: `dataset/font_dataset.py` - Updated `get_path()` method

3. **Training Now Works End-to-End**:
   * ✅ Training successfully starts
   * ✅ Progress tracking working correctly (real-time loss and iteration counts)
   * ✅ Loss decreasing as expected
   * ✅ Training completes successfully

**Test Results (2026-02-24 14:00-14:24)**:
- Started training with 30 processed characters
- ✅ **TRAINING COMPLETED SUCCESSFULLY** 
- Total duration: ~24 minutes (1500 steps)
- Final Loss: 0.168 (excellent! decreased from 0.6+)
- Speed: ~1.5 steps/second (consistent)
- Progress tracking: Working perfectly, real-time updates
- UI display: Clear and accurate
- Reference images: Successfully backed up
- Training state persists correctly: `State: completed, Progress: 100%`

### ✅ Previous Session Achievements (2026-02-23)
1. **Logo Enhancement**:
   * Changed mohen.svg color to light blue (#00A3FF) for proper visibility in dark theme
   * Enlarged logo from 32px to 64px (w-5 h-5 → w-8 h-8) with proportional container scaling
   * Removed outer border and background styling for cleaner aesthetic
   * Applied changes to all 4 HTML templates (step1, step2, step3, generate)

2. **Header UI Cleanup**:
   * Removed unnecessary user icon and left divider (marked as `hidden`)
   * Simplified header to focus on core navigation elements

3. **Training Complete UX Improvement**:
   * Added "Back to Save" button in terminal view for Step 3 (training completion screen)
   * Users can now easily toggle between viewing training logs and saving their font style
   * When closing logs, "Back to Save" button appears to return to the save panel without losing progress

### ✅ Previous Session Achievements
1. **Git Initialization & Cleanup**:
   * Removed old `.git` history to start fresh
   * Deleted temporary scripts and user data
   * Configured `.gitignore` properly
   * Initialized repository and created Initial Commit

2. **Remote Configuration**:
   * Added remote origin: `https://github.com/wrongre/Mohen`
   * Successfully pushed initial commits to GitHub

3. **Documentation Integration**:
   * Merged README, requirements, and .gitignore from doc/ folder
   * Updated requirements.txt with current dependencies

### 🚧 Current Issues & Backlog

#### 1. ✅ FIXED (This Session)
*   ~~**Startup Connection Error**~~ - Fixed with intelligent retry logic
*   ~~**Training Pipeline Failure**~~ - Fixed `.gitkeep` handling in dataset loader

#### 2. Immediate Tasks (Next Session)
*   **Training Completion Testing**: Monitor full training cycle to ensure consistent success
*   **Core Quality**: Implement Adaptive Thresholding in `utils_slice.py` to reduce background & grid artifacts
*   **Text Alignment**: Improve `ttf2im` centering logic using font metrics
*   **Generate Variable Slider**: Debug & verify if variation slider actually affects output

#### 2. Feature Development (Next Phase)
*   **UI Comprehensive Review**: Screen-by-screen review for additional refinements and consistency
*   **Chinese Localization**: Prepare multilingual support (UI text, prompts, messages)
*   **Training Parameter Tuning**: Increase cloning accuracy - currently insufficient similarity. Adjust diffusion steps, learning rates, model precision
*   **Character Limit Expansion**: Increase limit from current 20 characters to support full font families
*   **Flow Utilization**: Currently only slicing user's input - need to leverage user's flow patterns more effectively in training
*   **Generate Variable Control**: Investigate why slider adjustment shows minimal/no visual effect (may be too subtle or parameter needs scaling adjustment)

#### 3. Core Quality (Technical Debt)
*   **Background & Grid Artifacts**: Implement Adaptive Thresholding in `utils_slice.py`
*   **Alignment**: Improve `ttf2im` centering logic using font metrics

## 📋 Next Session Plan (Tonight - 2026-02-24 Evening)

**Priority 1: Git Cleanup & Commitment**
*   Remove untracked test data from working directory
*   Commit 2 bug fix commits to main
*   Push to GitHub

**Priority 2: Feature Development (If Time)**
*   Monitor if training persistence works (load completed model)
*   Test generation with completed model
*   Verify "save font" functionality for completed training
*   Review AI_CONTEXT.md backlog items from previous session

**Priority 3: Next Technical Tasks**
*   **Core Quality**: Implement Adaptive Thresholding in `utils_slice.py` to reduce background & grid artifacts
*   **Text Alignment**: Improve `ttf2im` centering logic using font metrics
*   **Generate Variable Slider**: Debug & verify variation slider effect

## 📋 Next Session Plan

**Priority 1: UI Comprehensive Polish**
*   Full screen-by-screen review
*   Identify and fix any remaining UI/UX inconsistencies
*   Prepare for Chinese localization

**Priority 2: Feature Enhancement**
*   Chinese localization implementation
*   Parameter tuning for better cloning accuracy
*   Variable slider debugging

## 🛠 Command Reference
*   **Start App**: Double-click `start_app.bat` or `python -u web_ui/main.py`
*   **Git Status**: `git status`
*   **Git Commit**: `git add . && git commit -m "message"`
*   **Git Push**: `git push origin main` (batch push after daily debugging)
*   **Stop Server**: Close terminal or Ctrl+C

## 📝 Current Session Git Changes

**Modified Files** (Ready to commit):
- `start_app.bat` - Added intelligent server readiness detection with `-UseBasicParsing` flag
- `dataset/font_dataset.py` - Fixed `.gitkeep` handling in `get_path()` method to skip non-directory files
- `AI_CONTEXT.md` - Updated with bug fix details, test results, and session status
- `.gitignore` - (Minor changes)

**Untracked Files** (To be cleaned before commit):
- `data_examples/processed/*.jpg` - Test character images (30 files)
- `data_examples/processed/grid_warped.jpg` - Test grid image
- `data_examples/train/TargetImage/MyStyle/` - Test directory
- `data_examples/upload/*.jpg` - Uploaded test file

**Commits to Make** (Tonight):
1. "Fix startup connection error with intelligent retry logic"
2. "Fix training dataset loader to skip non-directory files (.gitkeep)"

**Status**: Verified end-to-end training pipeline working correctly. Ready to clean up and commit.

---

## 📝 Session Log — 2026-03-05

Summary of today's work:
- 启动并调试 `Generate` 页面，集成本地 `makemeahanzi` 数据源；后端提供 `/api/stroke_order` 与 `/api/stroke_flow_run` 接口供前端查询与批量运行。
- 在前端实现格内逐笔显现动画并将右下预览与格内播放同步；尝试为大预览加入笔迹绘制（stroke-dashoffset）动画，但该改动导致大字显示顺序不对，用户要求回滚。
- 已回滚 `animatePreviewStroke` 并恢复使用 `renderStrokeFrame(item,index)`，服务器已重启并验证响应正常。

Current state (end of day):
- Backend: `makemeahanzi` 适配器可用，缓存与清理端点存在，`/api/stroke_flow_run` 返回每字笔画数据。
- Frontend: 写字格逐格播放功能正常；大预览已恢复为原始逐笔显示（未绘制动画）。
- Pending: 如需重做绘制动画，需先修正笔画顺序与同步逻辑或直接使用服务器端顺序保证一致性。

Actions performed now:
- 将本次会话记录写入 `AI_CONTEXT.md` 并提交到本地仓库。
- 已准备把修改推送到远程 GitHub 仓库（稍后执行 push）。

End of log.
