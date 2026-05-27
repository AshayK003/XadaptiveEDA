# Engineering Memory

High-signal heuristics extracted from real work. Updated after meaningful tasks. Outdated approaches are replaced, not appended.

---

## Architecture Patterns

### Dispatch Dict over if/elif Chains
Use `{type: handler_fn}` dicts for routing (e.g., visualization dispatch). Adding a new type requires one entry in the dict and one handler method — no touching `if/elif` chains. Pattern: `handler = dispatch.get(type); return handler() if handler else fallback`.

### Session State Initialization Gate
In Streamlit, gate every session key behind `if key not in st.session_state`. One block, all keys, at the top of the file. Never inline checks — they create subtle init-order bugs. Clone mutable defaults with `.copy()` to prevent cross-user leakage.

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

---

## Performance Optimizations

### Eliminate Redundant Rendering Libraries
One charting library per app. matplotlib + seaborn + plotly together adds 3 dependency trees, conflicts in render backends, and confuses users. Choose plotly for interactivity, remove the rest.

### Empty Dataset Guard First
Check `df.empty` at the top of `profile_dataset()` and return a skeleton schema. Prevents division-by-zero on `missing_percentage`, crashes on `.select_dtypes()`, and confusing errors downstream. Every processing function should have an empty-input path.

### Fix Thresholds over Continuous Computation
Preference weights: clamp + delta beats normalization every time. Normalization creates drift where all weights shift when only one changes. Fixed clamping (`max(0.1, min(1.0, current + delta))`) is simpler, predictable, and debugable.

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
|---|---|
| Run app | `streamlit run app.py` |
| Install deps | `pip install -r requirements.txt` |
| Git add+commit | `git add -A; git commit -m "prefix: message"` |
| Push to origin | `git push -u origin main` |
| Sync second copy | `Copy-Item -Path "D:\...\file" -Dest "C:\...\file" -Force` |

---

## Framework Insights

### Streamlit
- `st.experimental_rerun()` → `st.rerun()` in ≥1.36.0
- `st.info()`/`st.success()` for persistent messages; `st.toast()` for transient feedback
- `st.form()` + `st.form_submit_button()` prevents slider-change rerenders
- Session state keys persist across reruns but NOT across page reloads
- Widget keys must be globally unique — collisions produce silent cross-talk

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
- No normalization: weights are independent, one change doesn't shift others
- History: append-only list of dicts with type, action, timestamp
- Merge: external slider values overwrite, not accumulate

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
