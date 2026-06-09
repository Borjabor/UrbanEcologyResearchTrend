# Project Feedback: Urban Ecology Research Trends

*Reviewed 2026-06-09 by Opus 4.8 — holistic pass over `streamlit_app.py`, the analysis notebook, the data pipeline, and the API/methodology choices.*

## Executive summary

This is a genuinely good project: the data source is the right one, the UI is polished, and the regression code shows real statistical care. But a whole-project read surfaces four things worth acting on, roughly in priority order:

1. **The code has drifted from its own documentation.** Three items this file previously marked "✅ Fixed" are *not* fixed in the current `streamlit_app.py` — including the `set_page_config` ordering bug and the cached-client pattern. Either a revert happened or the fixes never landed. The docs (this file + `CLAUDE.md`) now describe code that doesn't exist.
2. **The headline statistical comparison is methodologically invalid as written.** Comparing a linear model's R² against an exponential model's R² the way the app does is not a valid model-selection procedure — the two R²s live on different scales.
3. **The app loads and computes a large amount of data it never uses**, slowing the first load for no benefit.
4. **The keyword methodology has validity limits that aren't disclosed** — worth a sentence in the README rather than a code change.

The API choice itself (OpenAlex) is **not** a weakness — see below. The weaknesses are in how the data is queried and how the trends are compared, not in where it comes from.

---

## What's working well

**OpenAlex was the correct API choice.** With Microsoft Academic Graph dead and Semantic Scholar's free tier now rate-limited, OpenAlex is the strongest free scholarly source available: 250M+ works, no API key, cursor pagination, and a "polite pool" via `mailto`. The pipeline uses all of these correctly (`per_page=200`, cursor paging, `mailto` set). The resumable harness — per-keyword cursor checkpointing plus `done_*.txt` markers and a 2-attempt retry with backoff — is a thoughtful, production-minded touch for a long scrape. Don't switch APIs.

**Linear regression is done properly.** Slope confidence intervals, R², p-value, and standard error are all computed and surfaced. The exponential fit converts the log-space slope back to an interpretable annual growth rate and doubling time.

**Real fixes that did land** (verified in current code): the geographic tab now derives every chart from a `df_countries_geo` base pre-filtered by year and keyword; `df_totals` counts distinct `paperId` per year via `nunique`, so multi-keyword papers aren't double-counted in the Total line; the `log(y+1)` offset was removed from `log_trend_analysis` in favor of excluding zero-count years; the hardcoded `66000` estimate was replaced with a real `count='exact'` HEAD query; and `display_chart_control` no longer nests its own columns.

**Caching, UI, credentials, README-as-research-output** — all still strong, as previously noted. `.env` and `*.db` are correctly git-ignored.

---

## 1. Critical: documentation/code drift

These three are documented as fixed but are live in `streamlit_app.py` right now:

**1a. Module-level Supabase client runs before `st.set_page_config` — and is dead code.**
`streamlit_app.py:59` calls `supabase: Client = get_supabase_client()` at import time, *before* `st.set_page_config(...)` at line 61. `get_supabase_client()` can call `st.error()`/`st.stop()`, which Streamlit forbids before `set_page_config`. Worse: the `supabase` variable it assigns is **never referenced anywhere** — every query calls `get_supabase_client()` fresh instead (lines 123, 201). So this line is pure dead code that only exists to trip the ordering bug. Delete line 59 outright.

**1b. `get_supabase_client()` is not cached, so a new client is created on every call.**
This file and `CLAUDE.md` both claim it's decorated with `@st.cache_resource`. It isn't (`streamlit_app.py:29`). During a cache-miss load, `get_all_papers_paginated` calls it once per page (line 123) — ~14 client constructions for a 66K-row load. Add `@st.cache_resource` as documented; that also makes the lazy-init pattern actually work.

**1c. `ORIGIAL_KEYWORDS` typo; docs say `URBAN_KEYWORDS`.**
The constant at `streamlit_app.py:24` is misspelled `ORIGIAL_KEYWORDS` and referenced as such (lines 262, 782). The consolidation to a single definition did happen — good — but the name doesn't match `CLAUDE.md`'s `URBAN_KEYWORDS`. Rename for correctness and to match the docs.

> Recommendation: after fixing these, re-verify every "✅ Fixed" claim against the actual file before re-marking it. The drift suggests fixes were made on a branch/notebook that didn't reach `streamlit_app.py`.

## 2. Statistical / analytical methodology

**2a. Linear-vs-exponential model selection by raw R² is invalid.**
`compare_linear_vs_exponential` (`streamlit_app.py:436`) picks the "better" model with `r_squared_log > r_squared_linear`. But `r_squared_linear` is computed on raw counts `y`, while `r_squared_log` is computed on `log(y)` — the regression in `log_trend_analysis` fits `linregress(years, np.log(paper_counts))` (line 396–398). **R² values from fits on different response scales are not comparable.** A log-scale fit almost always reports a higher R² simply because logging compresses variance, which is exactly why the exponential model "wins" nearly everywhere. This is the single most important analytical issue, and it's present in both the notebook and the app.
Fix options, best first:
  - Compare **AIC/AICc** computed in a common space (back-transform the exponential predictions to count space, then compute residual sums of squares for both models there).
  - Or compute the exponential model's R² on **back-transformed predictions** (`exp(fit)` vs actual counts), so both R²s are in count space.
  - Either way, also report residual diagnostics — exponential growth in raw counts produces heteroscedastic residuals that a single R² hides.

**2b. Leftover `- 1` in the exponential prediction curve.**
At `streamlit_app.py:1121`: `log_pred = np.exp(log_kw['slope'] * year_range_extended + log_kw['intercept']) - 1`. The `- 1` is a holdover from the old `log(y+1)` transform. Since `log_trend_analysis` no longer adds 1 before fitting, the fitted model is `count = exp(intercept + slope·year)` and subtracting 1 makes the **plotted curve inconsistent with the R² and growth rate reported beside it**. Remove the `- 1`.

**2c. Counts aren't normalized against global publication growth.**
The *entire* scientific literature grows ~exponentially (a few % per year). So "urban ecology grows exponentially at 11%/yr" partly just restates the base rate of all science expanding. The notebook handles this well with a general-ecology control — but **that control never made it into the app**. Consider a "share of all literature" view (urban-ecology papers ÷ total OpenAlex papers that year) or porting the general-ecology comparison line. This would turn "research is growing" into the much stronger "research is growing *faster than the field as a whole*."

**2d. `search=` is broad full-text matching → noisy attribution.**
The scrape uses OpenAlex `search=<keyword>` (notebook cell 9), which matches across title, abstract, and indexed full text. A paper that mentions "urban wildlife" once in a discussion section is counted as an "urban wildlife paper." This inflates counts and blurs keyword boundaries. More precise options: OpenAlex **Topics/Concepts** filters, or `title_and_abstract.search`. At minimum, **disclose** in the README that counts reflect keyword *mentions*, not curated topic membership.

**2e. First-author-country-only geographic attribution.**
`firstAuthorCountryIso` attributes each paper to one country. For a heavily collaborative, internationalized field this undercounts contributing nations and biases toward whoever is listed first. Fractional counting across all affiliations (each paper contributes 1/N to each country) would be more representative. If keeping first-author attribution, note the limitation on the map.

## 3. App correctness & performance

**3a. `df_expanded` is built with a 66K-row `iterrows` loop and never used.**
`load_data` constructs `df_expanded` (`streamlit_app.py:258–273`), returns it (306), and unpacks it in `main` (506) — but nothing in any tab reads it. The only purpose `title`, `authors`, and `citationCount` serve is to populate this dead frame. **Drop those three columns from the `select` (line 214) and delete the `df_expanded` construction.** That removes a per-row Python loop and a large network/memory cost (title + author strings × 66K rows) from every cold load.

**3b. "Countries Analyzed" metric is wrong.**
`streamlit_app.py:837`: `st.metric("Countries Analyzed", len(df_countries_filtered))`. `df_countries_filtered` is grouped by `(country, alpha3_code, year)`, so `len(...)` counts country-year rows, not countries — it massively overstates. Use `df_countries_filtered['country'].nunique()`.

**3c. `st.plotly_chart(..., height=...)` is not a valid argument.**
Lines 677, 777, 829 pass `height=` to `st.plotly_chart`. Chart height belongs in `fig.update_layout(height=...)`; the kwarg is ignored (and is liable to raise on a future Streamlit). Set heights in the figure layout (as `display_chart_control` already does) and drop the kwarg.

**3d. Inconsistent expansion style.**
Keyword expansion is done with slow `iterrows` loops in `load_data` (lines 229–239, 259–271) but with vectorized `str.split().explode()` in the geographic tab (lines 786–788). Standardize on the vectorized form everywhere — it's faster and already proven in this codebase.

**3e. `citationCount` is fetched but never analyzed.**
Either drop it (see 3a) or use it — a citation-weighted trend, or a "high-impact papers over time" view, would be a cheap, compelling addition given the data is already there.

## 4. Code quality & hygiene

- **`load_data` UI side-effects inside a `@st.cache_data` function** — still true, still minor. The placeholders are cleared on the cache-miss path, so the practical risk is low; the cost is testability. Splitting the pure data fetch from the progress UI would let you unit-test the pipeline.
- **Commented-out "Clear Cache" button** (`streamlit_app.py:526`) — restore it as a debug affordance or delete the comment.
- **`.DS_Store` is tracked in git** — add it to `.gitignore` and `git rm --cached` it.
- **No tests anywhere.** The regression functions (`linear_trend_analysis`, `log_trend_analysis`, `compare_linear_vs_exponential`) are pure and easy to test — a handful of fixtures with known slopes would protect against the kind of drift described in §1 and catch the §2a/§2b math issues.
- **README typos** (research-facing): "Stramlit", "Relatioships", "conslusion", "waas", "looked more ate the field". Quick polish since the README is the public artifact.

## 5. Missing features / opportunities

- **Keyword co-occurrence / similarity tab.** The notebook's Ochiai/cosine co-occurrence analysis (cell 25) and its temporal version are arguably the most novel findings and aren't in the app at all. A "Keyword Relationships" tab would extend the dashboard meaningfully.
- **Urban vs. general-ecology control in the app** (see §2c) — the notebook has it; the app doesn't.
- **Normalized share-of-literature view** (see §2c).
- **Data freshness indicator** — still valid. Store a load timestamp alongside the cached data and show "Data last refreshed: X".
- **Extend past 2023.** The 1970–2023 cutoff was chosen because OpenAlex's 2024 data was incomplete at scrape time; that's now backfillable. A re-scrape would also let you adopt the Topics-based query from §2d.
- **Commit the SQLite → Supabase upload step.** The notebook only ever writes to local SQLite (`papers.db`, `ecology_papers.db`); nothing in the repo pushes that data up to Supabase, so the migration was a manual, unrecorded step. The Streamlit app reads from Supabase, but there's no committed code to *populate* it — meaning a re-scrape (see above) currently can't be pushed to production without repeating an undocumented process. Add a notebook cell at the end of Section 2 that loads `papers.db` and upserts into the Supabase `papers` table, e.g. `pd.read_sql('SELECT * FROM papers', conn)` → batched `supabase.table('papers').upsert(records).execute()` (chunk to ~500 rows/request, keyed on `paperId`). This closes the loop and makes the whole pipeline reproducible from the repo alone. The `DB_CONNECTION_STRING` already in `.env` is presumably the leftover from however this was done by hand.

## 6. Unnecessary / verbose / reinvented code

A pass looking specifically for redundant logic and hand-rolled reimplementations of library built-ins (the kind of bloat AI-generated code tends to accumulate):

**6a. `linear_trend_analysis` recomputes the slope standard error that `linregress` already returns.**
`streamlit_app.py:328` captures `std_err` from `stats.linregress(...)` — that value *is* the standard error of the slope. Then lines 335–337 recompute the identical quantity by hand:
```python
s_yx = np.sqrt(np.sum((paper_counts - (slope * years + intercept))**2) / (n - 2))
s_xx = np.sum((years - np.mean(years))**2)
slope_se = s_yx / np.sqrt(s_xx)
```
`slope_se` is mathematically equal to the `std_err` already in hand, so the confidence interval can just use `std_err` directly (which is what the notebook version does — the app drifted into the more verbose form). Delete the three lines and use `slope - t_val*std_err` / `slope + t_val*std_err`.

**6b. The country treemap is hand-built with `go.Treemap` and manual id/label/parent lists.**
Lines 730–762 manually assemble `labels`/`ids`/`parents`/`values`/`colors` arrays plus a synthetic root node to feed `go.Treemap`. Plotly Express does the whole thing from the dataframe in one call: `px.treemap(df_top_country_totals, path=[px.Constant('All Papers'), 'country'], values='total_count', color='total_count', color_continuous_scale='Viridis')`. That replaces ~35 lines (including the `colors = [np.nan] + values[1:]` trick and `branchvalues='total'` bookkeeping) with one, and the root/branch totals are handled automatically.

**6c. The Time Series summary duplicates the same filter-and-aggregate block four times.**
In tab 2 (`streamlit_app.py:885–919`), `col1` and `col2` each contain a near-identical `if "Total (All Keywords)"… else…` block that re-derives the same filtered `df_totals` slice, just to take `.sum()` in one and `.mean()` in the other. Compute the filtered frame **once** above the columns, then read `.sum()` and `.mean()` off it. Removes ~25 lines and the risk of the two branches drifting.

**6d. `get_country_name` is cached, but it wraps an in-memory dict lookup.**
`@st.cache_data(ttl=3600)` on `get_country_name` (`streamlit_app.py:167`) adds Streamlit's hashing + serialization overhead around a `pycountry.countries.get(alpha_3=...)` call that is already an O(1) in-memory lookup — the cache machinery costs more than the thing it caches. There are only ~200 country codes; either drop the decorator, or (better) build the alpha-3→name mapping once with a dict/`.map()` over the column instead of calling a function per row. The `except (KeyError, AttributeError)` is also near-dead: `pycountry.countries.get` returns `None` rather than raising, and the `if country else` already handles that.

**6e. `get_supabase_client` has dead initialization and redundant nesting.**
`streamlit_app.py:34–35` sets `url = None; key = None` immediately before overwriting both with `os.getenv(...)` — dead lines. The function also wraps a try/except around code whose only failure modes it re-reports generically. It can collapse to roughly: read env vars, fall back to `st.secrets`, `st.stop()` with a message if missing, else `return create_client(url, key)`. (Fold this together with the `@st.cache_resource` fix from §1b.)

**6f. Notebook: `save_paper_to_db` / `save_author_to_db` open a new SQLite connection per row.**
In notebook cell 5, `save_paper_to_db` calls `sqlite3.connect(PAPERS_DB_PATH)` on *every* paper — so the 66K-paper scrape opens and tears down ~66K connections, each doing its own `SELECT`-then-`INSERT/UPDATE` round trip. Open one connection for the whole loop (or use `INSERT … ON CONFLICT` / `executemany`) and pass it in. This is the single biggest avoidable cost in the ingest pipeline. Same pattern in the author loop.

**6g. Notebook co-occurrence counting is hand-rolled with nested dicts.**
Cell 27 builds `year_pair_counts` as nested dicts via `iterrows` + manual key initialization, then converts back to a DataFrame to pivot. `itertools.combinations` into a `Counter`, or an `explode` + self-merge + `groupby`, expresses the same result far more compactly and faster. Lower priority (exploratory notebook code), but it's the same verbosity pattern as the app's `iterrows` loops (§3d).

---

## Suggested priority order

| # | Item | Effort | Why it matters |
|---|------|--------|----------------|
| 1 | Delete dead module-level client (§1a) | trivial | Active `set_page_config` bug |
| 2 | `@st.cache_resource` on client (§1b) | trivial | Many redundant connections per load |
| 3 | Fix model-selection math (§2a) + drop `-1` (§2b) | medium | The headline analysis is currently unsound |
| 4 | Drop unused columns + `df_expanded` (§3a) | low | Faster cold load, less memory |
| 5 | Fix "Countries Analyzed" metric (§3b) | trivial | Visibly wrong number |
| 6 | Rename `ORIGIAL_KEYWORDS` (§1c), remove `height=` kwargs (§3c) | trivial | Correctness/forward-compat |
| 7 | Cleanup pass: reuse `std_err` (§6a), `px.treemap` (§6b), de-dupe TS summary (§6c) | low | Removes ~70 lines of avoidable bloat |
| 8 | One-connection SQLite ingest (§6f) | low | Biggest avoidable cost in the scrape pipeline |
| 9 | Add general-ecology / normalized view (§2c) | medium | Turns the finding from weak to strong |
| 10 | Keyword-relationship tab (§5), tests (§4) | medium | Depth + regression protection |
