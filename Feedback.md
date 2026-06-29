# Project Feedback: Urban Ecology Research Trends

*Reviewed 2026-06-11 by Opus 4.8 — second full pass over the rebuilt notebook, `streamlit_app.py`, both databases, and the live OpenAlex API. Supersedes the 2026-06-09 review; resolved items are listed once and removed.*

## Executive summary

The overhaul work since the last review is genuinely strong: the NLS/AIC model comparison is methodologically correct, the scrape harness is cleaner, and most of the app's trivial bugs landed. But this pass found **one critical data bug that invalidates the rebuilt corpus**, and it has to be fixed before anything downstream matters:

1. **🚨 The re-scrape's type filter silently excluded all journal articles.** The corpus you've been analyzing is books and book chapters only (~25% of the intended literature). Every new finding — growth rates, the lin-vs-exp flips, country coverage — was computed on the wrong corpus. See §1.
2. **The notebook's statistics are now sound in structure** — count-space NLS + AIC was the right fix. A few refinements would make the conclusions more honest about uncertainty (§2).
3. **The app port is still pending** — `streamlit_app.py` retains the old invalid R² comparison and the other items already catalogued (§3).
4. **Narrative/docs drift**: several notebook markdown cells and the README still describe the old method and the old 2023 cutoff (§4).

---

## What's fixed and verified ✅

- **App**: `@st.cache_resource` on `get_supabase_client()`; `set_page_config` ordering; dead module-level client deleted; `URBAN_KEYWORDS` renamed; "Countries Analyzed" uses `nunique()`; invalid `height=` kwargs removed; page limit derived from `total_count`; `get_supabase_client` body de-cluttered.
- **Notebook**: `compare_linear_vs_exponential` now fits the exponential by **NLS in count space** (`curve_fit`, `y = a·exp(b·t)`) and selects by **AIC computed on count-space RSS for both models** — the cross-scale R² fallacy is gone. Residual diagnostics panel added (cell 32, plot 4). The old `log(y+1)` bias is gone.
- **Scrape**: single SQLite connection reused for the whole paper scrape, committed per page (§6f fixed for papers); `title_and_abstract.search` replaces broad full-text `search=`; general-ecology control fetched with the identical method and made disjoint at count time (urban paperIds excluded, DB left intact).
- **Co-occurrence pair counting** now uses `itertools.combinations` + `Counter` (cell 26).

---

## 1. 🚨 Critical: `type:journal-article` matches **zero** works — the corpus is books-only

The scrape filter (cell 9) is:

```
type:journal-article|book|book-chapter|monograph
```

OpenAlex renamed its types years ago: journal articles are `article`, and `monograph` is not a type at all. OpenAlex does **not** error on unknown filter values — it silently returns zero matches for them. Verified live (2026-06-11):

| Filter for `title_and_abstract.search:urban ecology`, 1970–2025 | Count |
|---|---|
| `type:journal-article\|book\|book-chapter\|monograph` (as scraped) | **6,282** |
| `type:article\|book\|book-chapter\|monograph` (corrected) | 25,534 |
| no type filter | 29,428 |

The DB confirms it: `papers.db` has 6,284 "urban ecology" papers — exactly the broken-filter count. Spot-checking titles shows Cambridge UP eBooks and similar. **The 17,863-paper corpus contains only books and book chapters; ~75% of the intended literature (19,252 articles + 455 reviews for this keyword alone) is missing.** The ecology control (63,730) has the same bug, so the urban-vs-general comparison is at least internally consistent — but it's a comparison of *book* output, not research output.

**Consequences:** every headline number from the rebuild is unreliable — the 10.7%/yr vs 6.2%/yr urban-vs-general gap, the "urban ecology and urban biodiversity now read Linear" flip, the 70.5% country coverage (books have systematically worse affiliation metadata than articles), and the geographic rankings. Treat them all as TBD until re-scraped.

**Fix (exact edits):**

1. In the notebook's `scrape_papers` (the cell defining it in Section 2 — currently cell 9), change the filter line:
```python
# OLD (broken — journal-article and monograph match nothing):
'type:journal-article|book|book-chapter|monograph,'
# NEW:
'type:article|review|book|book-chapter,'
```
   Including `review` is standard for output trends (journal-published research); `monograph` is not a valid OpenAlex type. `preprint`/`dissertation` stay excluded deliberately — disclose that choice in the README.

2. Run the reset cell (the one that deletes `done_*.txt`/`cursor_*.txt` and both `.db` files — currently cell 7), then re-run the scrape cells for both `papers.db` and `ecology_papers.db`. Expect ~4× the volume (~70–90K urban papers); the author/ROR enrichment cells must be re-run too and will take proportionally longer. The `.db` files are git-ignored — nothing to commit there.

3. **Add a sanity-check cell** immediately after the scrape cells, so this class of bug can never silently recur. Ready to paste (uses only names already defined in the notebook):
```python
# Sanity check: DB counts must match OpenAlex's own totals per keyword.
# OpenAlex silently returns 0 for unknown filter values, so a typo'd filter
# shows up here as a massive mismatch instead of going unnoticed.
import requests, sqlite3

TYPE_FILTER = 'type:article|review|book|book-chapter'
for keyword in query_list:
    params = {
        'filter': f'title_and_abstract.search:{keyword},{TYPE_FILTER},publication_year:{YEAR_RANGE}',
        'per_page': 1,
        'mailto': 'andre.borja.miranda@gmail.com',
    }
    api_count = requests.get(BASE_URL, params=params, timeout=30).json()['meta']['count']
    with sqlite3.connect(PAPERS_DB_PATH) as conn:
        db_count = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE ',' || REPLACE(search_keyword, ', ', ',') || ',' LIKE ?",
            (f'%,{keyword},%',)
        ).fetchone()[0]
    ratio = db_count / api_count if api_count else float('inf')
    status = 'OK' if 0.95 <= ratio <= 1.05 else '*** MISMATCH ***'
    print(f'{keyword:25s} API: {api_count:6,d}  DB: {db_count:6,d}  ({ratio:.1%})  {status}')
```
   (Small drift is normal — OpenAlex updates continuously; flag only >5% deviation.)

> **Notes for whoever implements this:** cell numbers above refer to the current notebook order and may shift — locate cells by content, not index. After any programmatic edit to the `.ipynb`, reload the notebook from disk in Jupyter before running, and don't save a stale in-browser session over it. Don't touch the locked decisions: `title_and_abstract.search`, the 6 keywords, year range 1970–2025, first-author geo attribution, NLS/AIC model selection, disjoint control.

> ✅ **Status 2026-06-11: code edits applied (uncommitted).** A `TYPE_FILTER` constant now lives in the Section 1 setup cell, `scrape_papers` uses it (broken `journal-article|monograph` values removed), and the sanity-check cell is inserted right after the control fetch — it covers both `papers.db` and `ecology_papers.db` and asserts on >5% deviation. **Still to do (requires runtime): reload the notebook from disk, run the reset cell, re-run the scrape cells for both DBs, then the author/ROR enrichment cells, then the sanity-check cell.** All analysis numbers remain TBD until that completes.

## 2. Statistical methodology — sound core, four refinements

The NLS-in-count-space + AIC design (cell 32) is correct and well documented. Refinements, in value order:

**2a. Report ΔAIC honestly instead of a binary winner.** With k=3 for both models, the AIC contest reduces to comparing RSS, so `better_fit` flips on trivially small differences. Standard practice (Burnham & Anderson): |ΔAIC| < 2 means the models are statistically indistinguishable. You already compute `delta_aic` — add it to `display_cols` in the comparison printouts, and in `compare_linear_vs_exponential` change the verdict to a three-way label:
```python
if abs(aic_lin - aic_exp) < 2:
    better_fit = 'Indistinguishable'   # tie: report both, don't crown a winner
else:
    better_fit = 'Exponential' if aic_exp < aic_lin else 'Linear'
```
This matters because the lin-vs-exp "flips" are exactly the kind of finding a near-tie produces. Also switch `_aic_count_space` to AICc (correct at n≈56): `return aic + (2 * k * (k + 1)) / (n - k - 1)` appended to the existing return value.

**2b. The growth-rate comparison has no uncertainty statement.** Cell 34 concludes urban grows X points faster than general ecology, but both `b` coefficients come with standard errors you already compute (`std_error`, `slope_ci_lower/upper` in `log_trend_analysis` results). Print the two 95% CIs side by side, and add a formal test on the difference:
```python
b_diff = urban_log['log_slope'] - ecology_log['log_slope']
se_diff = np.sqrt(urban_log['std_error']**2 + ecology_log['std_error']**2)
p_diff = 2 * stats.norm.sf(abs(b_diff / se_diff))
print(f'Growth-rate difference: b={b_diff:.4f} (p={p_diff:.2e})')
```
If significant, "urban outpaces the field" becomes statistically defensible instead of a point-estimate comparison.

**2c. Recent-year indexing lag biases the growth comparison.** In the current DBs, general ecology *declines* after 2023 (3,472 → 3,251 → 3,268) while urban jumps +24% in 2025. OpenAlex indexing lags by 1–2 years, and the lag differs by publication type and field — so the most recent years can artificially inflate one side's growth rate. After the re-scrape, add a sensitivity check: refit excluding the last 1–2 years and report whether the urban-vs-general gap survives. The cell-2 comment "2025 is the last complete year" is optimistic — say "most recently *indexed*" and disclose the lag.

**2d. Known approximations worth one disclosure sentence each (no code needed):**
- Count-space NLS weights recent high-count years heavily; `b` is effectively a recent-decades growth rate. (This is the correct trade-off — you documented the converse log-fit bias well — just say it.)
- Gaussian equal-variance AIC is an approximation for count data whose variance grows with the mean. The canonical tool is a Poisson/negative-binomial GLM with log link (gets you exponential fit + valid AIC + count-appropriate variance in one shot via `statsmodels`). Optional upgrade, not required.
- Yearly counts are autocorrelated, so OLS/NLS p-values overstate significance. With p ≈ 10⁻²⁰ the conclusions don't change, but a one-line caveat is honest.

**2e. Minor plotting inconsistency:** cell 34's log-scale panel plots data as `log(counts + 1)` but the fitted line as `log_slope·year + log_intercept` (i.e., log(y)). For early low-count years the data points sit visibly off their own fit line. Drop the `+ 1` — the panel only plots years that exist in the data.

## 3. App port (`streamlit_app.py`) — the remaining bulk of work

Still on the old logic; port from the notebook once the corpus is rebuilt:

- **§2a-port**: replace `linear_trend_analysis` / `log_trend_analysis` / `compare_linear_vs_exponential` with the notebook's NLS/AIC versions (the app still selects by cross-scale R² at `streamlit_app.py:446` and reports "R² difference" as the comparison metric throughout tab 3, including the `better_model` pick at :1157 and the ΔR² verdicts at :1177–1186).
- **§2b-port**: leftover `- 1` in the exponential curve at :1112 — inconsistent with the fit plotted beside it.
- **§3a**: `df_expanded` (:251–266) is still built with a full-corpus `iterrows` loop, returned, and never read; `title`, `authors`, `citationCount` are fetched only to feed it. Drop all of it from the `select` (:207) for a faster cold load.
- **§3d**: `iterrows` expansion in `load_data` (:222–231) vs the vectorized `str.split().explode()` already used in tab 1 (:777–779) — standardize on the latter.
- **§6a**: `linregress` already returns the slope SE; delete the hand recomputation (:328–330) — the notebook version already does this right.
- **§6b**: hand-built `go.Treemap` (:718–768) → one `px.treemap(..., path=[px.Constant('All Papers'), 'country'], ...)` call (the notebook's cell 40 is the template).
- **§6c**: the four near-identical filter blocks in the Time Series summary (:876–910) → compute the filtered frame once, read `.sum()`/`.mean()` off it.
- **§6d**: `@st.cache_data` on `get_country_name` wraps an O(1) dict lookup; replace per-row `.apply` with one `.map()` over a prebuilt alpha3→name dict.
- **New-corpus constants**: the year slider and header text hardcode 1970–**2023** (:497, :531–535) → 2025; the "66,000+ papers" loading notice (:183) will be wrong for the new corpus; re-check after the re-scrape.
- **Port the new analyses**: urban-vs-general control line, the ΔAIC display (§2a), and ideally the keyword-similarity heatmap as a fourth tab — still the notebook's most novel content and still absent from the app.

## 4. Docs & narrative drift

- **Notebook markdown describes code that no longer exists**: cell 0 still says data is cut off at **2023**; cells 31 and 36 still describe the old "log transform to flatten the curve" method instead of NLS/AIC. Update alongside the §1 re-scrape.
- **README**: still claims across-the-board exponential growth, the 2023 cutoff, and the old corpus; typos remain ("Stramlit", "Relatioships", "conslusion"). Cell 29's markdown has the same two typos — fix both copies. Hold the rewrite until the corrected corpus numbers exist, then do one pass.
- **Mutation across cells**: cell 18 renames `df_totals['search_keyword']` to `'Urban Research'` in place, so cell 32's results silently depend on cell 18 having run. Build `df_combined` from a copy instead — this is exactly the out-of-order-execution trap this project has hit before.

## 5. Code hygiene

- **§6f is only half-done**: `save_author_to_db` (cell 4) still opens a connection per author, and cell 12's `had_institution_before` check opens *another* connection per author inside the batch loop. Same one-connection fix as the paper scrape; with a 4× corpus this loop gets 4× slower.
- **`ROR_CLIENT_ID` is hardcoded in cell 2** and committed. ROR client IDs are low-sensitivity rate-limit tokens, but the project already has an `.env` pattern — move it there for consistency.
- **`.DS_Store` is still tracked** (`git ls-files` confirms). `git rm --cached .DS_Store` + gitignore entry.
- **No tests.** The regression functions are now pure and stable in shape — fixtures with a known linear series, a known exponential series, and a near-tie would lock in the §2a behavior and protect the app port.
- **Cell 17/28** still use `iterrows`/nested-dict expansion; cell 26 fixed the pattern, these two didn't. Low priority.
- **Missing: committed SQLite → Supabase upsert** (cell 45 is empty). Still the gap that makes production data a manual, unrecorded step — and it's now blocking, since the re-scraped corpus must reach Supabase for the app to use it. Batched `upsert` keyed on `paperId`, ~500 rows/chunk.

## 6. Visuals

The dashboard and notebook figures are in good shape (consistent Viridis/dark theme, sensible chart types, residual panel added). Worthwhile polish:

- **Log-scale toggle on the Time Series tab.** For data whose headline claim is exponential growth, a `st.toggle` → `fig.update_yaxes(type='log')` lets users *see* the straight line. Cheapest high-value visual addition.
- **The Top-N country column renders up to 30 separate Plotly figures** (:675–705) — noticeably heavy. A single faceted `px.line(..., facet_row='country')` or `st.line_chart` sparklines would cut render time and DOM weight substantially.
- **Annotate the 2020 COVID dip** on the time-series chart (`fig.add_annotation`) — the notebook narrative calls it out; the app chart should too.
- **`st.subheader` used for a paragraph** of body text under the title (:497) — semantically a `st.markdown`, and it renders oversized.
- The animated bar chart (cell 43) is a nice notebook touch; it would port to the app well if you want a "wow" element in the Geographic tab.

---

## Suggested priority order

| # | Item | Effort | Why it matters |
|---|------|--------|----------------|
| 1 | ~~Fix type filter~~ ✅ done · **re-scrape both DBs** (§1) | hours of runtime | Current corpus is the wrong literature; everything downstream is provisional |
| 2 | ~~Add scrape-vs-API count sanity check~~ ✅ done — run it after the re-scrape (§1) | — | Would have caught #1; prevents recurrence |
| 3 | Re-run notebook on corrected corpus; ΔAIC ties + growth-rate CIs + lag sensitivity (§2a–c) | low | Findings become defensible, not just directional |
| 4 | Supabase upsert cell (§5) | low | Blocking: new corpus must reach the live app |
| 5 | App port: NLS/AIC + `-1` + year constants (§3) | medium | The deployed analysis is still the invalid one |
| 6 | App cleanups: df_expanded, iterrows, treemap, summary de-dupe (§3) | low | Faster load, ~100 fewer lines |
| 7 | One-connection author loop (§5) | low | 4× corpus makes this loop the new bottleneck |
| 8 | README + notebook markdown refresh (§4) | low | Public artifact; currently describes the old project |
| 9 | Log-scale toggle, faceted country lines, COVID annotation (§6) | low | Best visual value per line of code |
| 10 | Tests for regression functions (§5) | medium | Protects the port and the §2a behavior |
