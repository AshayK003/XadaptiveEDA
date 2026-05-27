# Engineering Memory

High-signal heuristics extracted from real work. Updated after meaningful tasks. Outdated approaches are replaced, not appended.

---

## Architecture Patterns

### Dispatch Dict over if/elif Chains
Use `{type: handler_fn}` dicts for routing (e.g., visualization dispatch). Adding a new type requires one entry in the dict and one handler method — no touching `if/elif` chains. Pattern: `handler = dispatch.get(type); return handler() if handler else fallback`.

### Data Quality Pipeline Ordering
Cleanse pipeline order matters — deduplicate and normalize column names before running `select_dtypes` or column-name-based ops. If duplicate columns exist, `select_dtypes` returns a deduped frame that can't be assigned back by name. Correct order: normalize missing → dedup → normalize names → inf/nan → remove empty cols → remove empty rows → detect/flag → infer types.

### Missing Token Normalization
Convert common missing tokens (`NA`, `N/A`, `NULL`, `""`, `"-"`, `"?"`, `"#N/A"`) to `NaN` with a `frozenset` lookup on string-stripped values. Apply only to `object`-dtype columns to avoid unnecessary overhead on numeric columns. Use `pd.api.types.is_object_dtype()` as the gate.

### Type Inference with Confidence Thresholds
Never infer types on the full dataset in one shot — sample first. Use 90% numeric parse ratio as the threshold for casting to numeric. For datetime, require a column-name hint (`date`, `time`, `timestamp`) AND 80% parse ratio. This prevents corrupting ID fields or mixed-type columns.

### Quality Score as Weighted Composite
Score = 0.3×completeness + 0.2×uniqueness + 0.15×no_duplicates + 0.15×no_sparse + 0.10×no_constant + 0.10×no_mixed. Each sub-score is a 0–1 ratio with optional capping (`min(1 - X / threshold, 0.5)`). Display with 🟢(≥0.8) / 🟡(≥0.5) / 🔴(<0.5) indicator.

### Session State Initialization Gate
In Streamlit, gate every session key behind `if key not in st.session_state`. One block, all keys, at the top of the file. Never inline checks — they create subtle init-order bugs. Clone mutable defaults with `.copy()` to prevent cross-user leakage. Keep separate toggles for independent features (ai_enabled vs _chat_enabled) — bundling them creates user confusion.

### Shared Constants File
Extract strings and literals that appear in ≥2 modules into a single `constants.py`. Includes type lists, default weights, UI labels. Single source of truth prevents drift.

### Service Module Separation
Each module owns one responsibility: processing, scoring, tracking, generating, visualizing, constants. Streamlit file (app.py) is thin orchestration — init components, wire events, render UI. No business logic in app.py.

---

## Debugging Strategies

### Check for Second Copies
When a user reports errors that don't match the codebase, check for stale project copies in other directories. Symptom: error mentions files/imports that were deleted from the main copy. Fix: copy-sync changed files to the stale copy.

### Isolate Error Format Inference
Format-inference fallback chains (`try xlsx except csv except json`) mask real errors. The final fallback swallows the root cause and produces a misleading message. Always let the appropriate parser error propagate directly.

### Deprecated API Sweep
When upgrading dependencies, grep for `experimental_`, `deprecated`, `FutureWarning` before touching anything else. These accumulate silently and break without warning.

### Bare except Catastrophe
`except:` catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`. Always use `except Exception:` unless you explicitly need to catch those. Sweep with regex `^\s*except\s*:`.

### Check .env is Actually Git-Ignored
`.env` in `.gitignore` is not enough — verify with `git check-ignore .env` and `git status --short .env`. A misplaced comment or CRLF issue can break gitignore patterns. Run a pre-commit hook that fails if any file matches known API key patterns (`gsk_`, `sk-or-v1`, `sk-ant`, `AIza`).

---

## Performance Optimizations

### Eliminate Redundant Rendering Libraries
One charting library per app. matplotlib + seaborn + plotly together adds 3 dependency trees, conflicts in render backends, and confuses users. Choose plotly for interactivity, remove the rest.

### Empty Dataset Guard First
Check `df.empty` at the top of `profile_dataset()` and return a skeleton schema. Prevents division-by-zero on `missing_percentage`, crashes on `.select_dtypes()`, and confusing errors downstream. Every processing function should have an empty-input path.

### Fix Thresholds over Continuous Computation
Preference weights: clamp + delta beats normalization every time. Normalization creates drift where all weights shift when only one changes. Fixed clamping (`max(0.1, min(1.0, current + delta))`) is simpler, predictable, and debugable.

### Inline Scaling over Full Recompute for What-If
Counterfactual sliders that preview ranking changes should scale scores inline (`new_score = old_score × (cf_value / old_pref)`) instead of re-running the full recommendation engine. Full recompute on every slider tick is O(n²) per tick — inline scaling is O(1).

### Capped LRU Cache in Session State
When caching LLM responses or other per-combination data in Streamlit session state, enforce a cap by scanning keys on each insertion. Pattern: `ai_cache_keys = [k for k in st.session_state if k.startswith('_ai_')]; if len(ai_cache_keys) > 20: del st.session_state[min(ai_cache_keys)]`. Without a cap, every unique column/type combination is cached forever.

### Pre-Read File to BytesIO Before Parsing
Before passing an uploaded file to pandas readers, read it into `BytesIO` for size checks and empty detection. This prevents zip bombs, extremely large files, and unreadable formats from hitting the parser. Also catches empty files early with a clear error instead of a pandas cryptic message.

### Dict Identity Preservation in Streamlit
When a method needs to replace all values in a dict that is referenced by both the class and `st.session_state`, use `.clear()` + `.update()` instead of reassignment. Reassignment breaks the shared reference and creates silent drift between the two copies. Pattern: `self.preference_weights.clear(); self.preference_weights.update(new_values)`.

### Skip Name Re-normalization After User Renames
If the user manually renames columns via UI text inputs, re-running `cleanse()` will undo those names (lowercase → spaces → underscores). Add a `skip_name_normalization=True` parameter to the pipeline that skips the `_normalize_column_names` step. Only re-cleans for quality metrics, not structure.

### LLM as Primary, NLP as Fallback for Classification
The NLQ classifier runs the LLM first (when AI is enabled) and falls back to the NLP engine only when the LLM is unreachable or returns low confidence. This gives better query understanding (synonyms, phrasing, context) while keeping the lightweight NLP path as a safety net. The NLP engine tokenizes, stems, expands synonyms, and scores by TF-weighted overlap — no external dependencies needed.

### Chat with Dataset Context
The `chat_with_data()` function builds a structured text summary of the dataset (shape, columns, numerical stats, categorical distributions, missing %, outliers, skewness, sample rows) and sends it as context with every user question. The last 6 message exchanges are appended for follow-up conversation. This avoids the need for retrieval-augmented generation (RAG) or vector databases for small-to-medium datasets — the full statistics fit in context.

### Separate Toggles for Independent Features
The AI insights toggle (`ai_enabled`) and the chat toggle (`_chat_enabled`) are independent sidebar toggles. They share the same provider/connection settings block (shown when either is enabled) but activate different UI sections. Users can enable chat without AI analysis (and vice versa). This avoids the common mistake of bundling unrelated features behind a single toggle.

---

## Failure Patterns

### Inflated Claims Become Tech Debt
Using "AI", "ML", "learning", "intelligent" as marketing for simple heuristics creates a trust problem when users probe. Worse: it becomes a maintenance trap when someone tries to actually add ML later and has to untangle the naming. Name things what they are.

### Lookup-Table "Insights" Are Brittle
Pre-written sentences keyed by analysis type always produce output regardless of actual data. Users notice when the "insight" is a generic sentence that doesn't match their dataset. Data-driven generation (reading actual profile values) is more work but always honest.

### Toggle Key Collisions in Streamlit
If two widgets share the same `key` (e.g., same analysis type rendered twice at different positions), toggling one affects the other. Always include a position index or unique ID in the key. Pattern: `f"{type}_{index}"`.

### Complex Preference Math Accumulates Bugs
The original code computed `percent_change` and stored it on the object but never used it. Dead code paths rot silently. Remove unused computations immediately — git history preserves them if needed later.

### Import Inside Render Loop
`from X import Y` inside a Streamlit render block adds import overhead on every rerun. Even though Python caches imports after the first time, the import machinery still fires a dict lookup. Always move imports to the top of the file. Pattern: grep for `import` and `from` statements at non-zero indentation.

### Broken Feature Paths Masked by Fallback
A feature path that calls a nonexistent method (`_call_llm` in nlq_engine.py) but has a fallback that works (keyword matching) is invisible in tests. The broken path never executes in practice but any future code change that tries to use it will crash. Always either implement the path or remove it — never leave a dead reference with a working fallback.

### Live API Keys in .env Must Be Rotated Immediately
If `.env` contains live API keys and is visible in any session output (error messages, debug logs, CI output), those keys are compromised. Rotate immediately — do not just delete the file. Keys may have been cached, logged, or exfiltrated.

---

## Refactoring Heuristics

### Dead Code Removal: Top-Down
1. Remove unused imports (compiler/mypy finds most)
2. Remove unused methods and their tests
3. Remove unused parameters and local variables
4. Remove unused modules
Each step may ripple, so batch in one commit per layer.

### Replace Pre-written Text with Template + Data
Instead of storing full sentences, store structure and format at call time. Example: InsightGenerator reads `data_profile` values and builds sentences with f-strings. Results change when data changes. Maintenance: update the template, not 50 canned sentences.

### Extract Constants Before Logic Refactors
Before touching any business logic, extract hardcoded strings, magic numbers, and type lists into a constants module. This creates clean diff boundaries: constants move in one commit, logic changes in the next. Reduces cognitive load per commit.

### Rename to Match Reality
When `PreferenceLearner` doesn't learn, rename it to `PreferenceTracker`. When `ExplainabilityEngine` doesn't explain, replace it. Names that over-promise create confusion for every future reader. Cost is small; payoff compounds.

---

## UX Patterns

### Empty States
Every data-dependent UI section needs an empty state. Three tiers:
1. **Caption**: `st.caption("Select columns...")` for optional controls with no selection
2. **Info box**: `st.info(...)` for pre-upload guidance (supported formats, limits)
3. **Welcome block**: full guidance section when no file is uploaded

### Metric Layout Stability
Use a fixed number of `st.metric` columns. Don't conditionally hide a metric — the gap breaks the visual flow. Show "0 (0.0%)" instead of suppressing the metric. Pattern: always compute the value, always render the metric.

### Feedback Without Page Flash
Button callbacks in Streamlit already trigger a rerun. Adding `st.rerun()` inside the callback causes a double-rerun flash. Remove it — just update session state. The toast persists through the natural rerun.

### Semantic Toggle over Button Toggle
`st.button` with manual session state management is a code smell. Use `st.checkbox` for binary state toggles:
- No manual session state needed (built-in)
- Screen reader announces checked/unchecked
- Label is visible static text, not a volatile button label
- No rerun just to toggle state (checkbox state changes trigger rerun, but don't need manual sync)

### Error Presentation
Top-level: polite message with file name and error type.
Expander: internal details (file path, traceback, Python executable).
Never show system paths or tracebacks as primary error message.

---

## Anti-Patterns

### AI/ML Labeling for Simple Logic
A weighted scoring formula is not machine learning. Fixed-delta adjustments are not reinforcement learning. Calling them that invites skepticism, misleads users, and makes the project look amateur. If it can be described in 2 sentences, it doesn't need a buzzword.

### Dead Imports in Requirements
Removing a library from the code but leaving it in `requirements.txt` wastes install time, bloats the venv, and causes confusion. Every dependency should have a corresponding import or documented indirect need.

### Format-Fallback Chains
Trying `pd.read_excel`, then `pd.read_csv`, then `pd.read_json` in sequence means the first error is swallowed and the last one (least likely to match) becomes the error message. Use file extension to pick the parser upfront.

### Pre-computed Insight Lookup Tables
Storing dicts of full-sentence insights keyed by analysis type is maintenance-heavy and produces wrong answers for edge-case data. Generate from current data profile instead.

---

## Tooling Shortcuts

| Context | Command |
|---|---|---|
| Run app | `streamlit run app.py` |
| Install deps | `pip install -r requirements.txt` |
| Run all tests | `python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py && python test_data_quality.py` |
| Syntax check | `python -m py_compile app.py` |
| Git add+commit | `git add -A; git commit -m "prefix: message"` |
| Push to origin | `git push -u origin main` |
| Sync second copy | `Copy-Item -Path "D:\...\file" -Dest "C:\...\file" -Force` |
| Check .env in git | `git check-ignore .env` |

---

## Framework Insights

### Streamlit
- `st.experimental_rerun()` → `st.rerun()` in ≥1.36.0
- `st.info()`/`st.success()` for persistent messages; `st.toast()` for transient feedback
- `st.form()` + `st.form_submit_button()` prevents slider-change rerenders
- Session state keys persist across reruns but NOT across page reloads
- Widget keys must be globally unique — collisions produce silent cross-talk
- `st.button` for toggles fires reruns on every click. Use `st.checkbox` instead — same visual footprint, no manual session state, accessible by default
- `st.rerun()` after a button callback is redundant — the button interaction already triggers a rerun. Double-rerun causes visual flash
- Not every metric needs to be conditional. Show "0 (0.0%)" instead of hiding the metric column entirely — keeps layout stable
- External links in the UI should not navigate away from the app. Replace sample dataset download links with inline info about supported formats
- `st.caption()` for empty states (e.g., "Select columns to generate a visualization") is lighter than `st.info()` and doesn't compete with errors/warnings
- Bar charts with flat data (all 0.5) are noise. Show only when data is loaded and preferences have diverged
- Error diagnostics should be structured: user-friendly message visible, `st.expander("Technical details")` with `st.code()` for internals. Never dump stack traces or file paths at top level
- Feedback buttons (👍/👎) benefit from including the recommendation title in the label for screen reader context across repeated instances
- `st.metric("Missing Values", ...)` should always render. A missing `st.metric` breaks the column layout — the remaining columns don't reflow properly

### Plotly
- `make_subplots()` + `add_trace()` for multi-pane charts
- `px` (plotly express) for quick single-chart, `go` (graph objects) for custom layouts
- All figures interactive by default (zoom, pan, hover, download)
- No need for `st.pyplot()`, use `st.plotly_chart()` (also renders faster)

### Pandas
- `.select_dtypes(include=['number'])` for numerical columns
- `.select_dtypes(include=['object', 'category'])` for categorical
- `.skew()` for distribution symmetry, `.isna().sum()` for null counts
- IQR outlier detection: `Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`
- `pd.read_excel()` needs openpyxl (`.xlsx`) or xlrd (`.xls`)

---

## Best Known Approaches

### Project Cleanup Order
1. Audit: read every file, map imports, identify dead code and inflated claims
2. Fix deprecations: update API calls to current versions
3. Remove dead code: imports → methods → modules
4. Extract constants: magic numbers and hardcoded strings → constants.py
5. Harden: add guards for empty/edge-case inputs, validate profile keys
6. Replace gimmicks: lookup tables → data-driven generation, fake AI → honest names
7. Sync deps: remove unused libraries from code and requirements.txt
8. Document: README with honest description, this memory file

### Preference/Tracking System Design
- All weights: fixed range `[0.1, 1.0]`, clamped after every adjustment
- Adjustments: symmetric fixed deltas (+0.10 liked, -0.10 disliked)
- Implicit actions: +0.05 explored, +0.03 column_selected, +0.03 viz_type_changed, -0.02 ignored
- History: append-only list of dicts with type, action, timestamp, metadata (optional)
- Metadata captured per event: column names, viz type, source event
- No normalization: weights are independent, one change doesn't shift others
- Merge: external slider values overwrite, not accumulate

### Recommendation Scoring Extended Schema
Each recommendation dict now includes full decomposition:
- `base_score`: 0.6–0.9 catalog value
- `pref_score`: user preference weight (0.1–1.0)
- `data_relevance`: data characteristic match (0.5–1.0)
- `quality_adjustment`: 0.5 + 0.5 × quality_score or None
- `diversity_penalty`: 0.85 if penalized, else None
- `score`: final composite after all adjustments
Columns sorted by interestingness descending (skew, outlier %, cardinality, missing %).

### Diversity Penalty
- Categories: numerical, numerical_pairs, categorical, datetime, any
- Second rec sharing same category gets 0.85× multiplier
- `any` category never penalized (missing_values)
- Applied after initial sort, then re-sorted

### Column Interestingness Scoring
Numerical: skew × 0.6 + outlier% × 0.4, capped at 1.0, penalized for missing.
Categorical: score peaks at cardinality ~10, ranges 0.3–0.9.
Missing: directly proportional to missing% (higher = more interesting).
Used to sort columns within each recommendation — interesting columns first.

### Global Explainability Dashboard
`global_explanation_summary()` in insight_generator.py returns markdown with:
- Dataset overview (rows × cols, quality score)
- Explored analysis types
- Feedback summary and preference weight table
- Data quality highlights (sparse, constant, mixed-type, duplicate counts)

### Progressive Sampling
Datasets >50k rows trigger a checkbox offering stratified sampling to 10k rows.
Sampled data is used for profiling, recommendations, and visualization.
Full dataset is NOT retained — sampling is permanent for the session.
Threshold: 50,000 rows. Target: 10,000 rows. Random seed: 42.

### Visualization Dispatch Pattern
```python
def generate_viz(self, analysis_type, df, columns, **kwargs):
    dispatch = {'type1': self._handler1, 'type2': self._handler2}
    handler = dispatch.get(analysis_type)
    return handler(df, columns, **kwargs) if handler else fallback_figure()
```
Each handler: returns `go.Figure()`. No shared state. `**kwargs` for type-specific options.

---

## Deprecated Approaches

| Replaced | With | Reason |
|---|---|---|
| 6-way if/elif visualization routing | dispatch dict | Single new-type addition, no chain edits |
| AI/ML labeling for heuristics | honest naming | Trust, maintenance, accuracy |
| ExplainabilityEngine (lookup sentences) | InsightGenerator (data-driven) | Always-relevant output, lower maintenance |
| PreferenceLearner with normalization | PreferenceTracker clamped deltas | Predictable per-weight behavior |
| matplotlib+seaborn+plotly | plotly only | 1 dep tree, interactive, faster render |
| format inference fallback chain | extension-based dispatch | Clear errors, no masked root causes |
| experimental_rerun | rerun | Deprecated in 1.36 |
| infer_datetime_format=True | let pd infer | Deprecated in pandas 2.0 |
| type inference on full dataset | sample-first with confidence threshold | Protects ID/mixed columns from corruption |
| inline missing-token handling | frozenset lookup on object cols only | Consistent behavior, no numeric-overhead |
| single-action implicit tracking | 5-action tracking with metadata | Richer feedback signal for explainability |
| raw column list in recs | interestingness-sorted columns | Best columns shown first to users |
| flat recommendation ranking | diversity-penalized ranking | Prevents same-category dominance |
| `infer_datetime_format=True` | `dayfirst=True` (dropping deprecated param) | Deprecated in pandas 2.0, removed in 3.0 |
| full-recompute counterfactual | inline score scaling | Avoids O(n²) per slider tick |
| unbounded AI response cache | LRU-capped 20-entry cache | Prevents session memory leak |
| re-cleanse after column rename | skip_name_normalization=True | Undoes user renames |
| inline import inside render block | top-level module import | Import overhead per rerun |
| load_data() reads raw file into pandas | pre-read into BytesIO for validation | Prevents zip bombs, empty files, garbled encoding |
| NLQ bar (text_input with NLP keyword matching) | dedicated chat interface (st.chat_input + LLM) | Chat provides better UX, handles any question, works with AI toggle |
