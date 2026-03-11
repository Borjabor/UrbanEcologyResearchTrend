# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (accessible at http://localhost:8501)
streamlit run streamlit_app.py

# Run the Jupyter analysis notebook
jupyter notebook UrbanEcologyResearchTrendAnalysis.ipynb
```

The app is also deployed at https://urbanecologyresearchtrend.streamlit.app/

To run from VSCode with the debug button, create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit",
            "type": "debugpy",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "${file}"],
            "justMyCode": true
        }
    ]
}
```

## Architecture

This is a single-file Streamlit app (`streamlit_app.py`) that visualizes urban ecology research trends (1970–2023) using data from a Supabase (PostgreSQL) backend populated via the OpenAlex and ROR APIs.

**Data flow:**
1. `get_supabase_client()` initializes the Supabase connection once via `@st.cache_resource` (lazy, called on first query). `st.set_page_config` is called first at module level before any Streamlit interaction.
2. `load_data()` runs a lightweight `count='exact'` HEAD query to get the exact paper count, then fetches all papers via pagination. Cached for 1 month (`@st.cache_data(ttl=2592000)`).
3. `df_totals` (the "Total (All Keywords)" series) is computed via `nunique` on `paperId` per year — multi-keyword papers are counted exactly once.
4. Data is filtered by user-selected keywords and year range in each tab before any aggregation.
5. Analysis functions compute regression models and geographic counts.
6. Plotly renders interactive charts in three tabs.

**Three UI tabs:**
- **Geographic Analysis** — Choropleth map, country time series, treemap, country×keyword heatmap. All charts are filtered by the sidebar year range, keyword selection, and top-N country count. A `df_countries_geo` base is derived from the raw `df_countries` at the top of the tab before any aggregation.
- **Time Series** — Publication counts over time by keyword. "Total (All Keywords)" shows unique paper counts with no double-counting.
- **Regression Analysis** — Linear vs. exponential growth model comparison with R² scores. `log_trend_analysis()` excludes years with zero paper count before fitting (no `log(y+1)` bias).

**Key functions in `streamlit_app.py`:**
- `get_supabase_client()` — `@st.cache_resource` singleton; returns the Supabase client, creating it only once
- `get_all_papers_paginated()` — paginates through the papers table; uses a `total_count` parameter (from the pre-fetch count query) for an accurate progress bar; page limit derived dynamically from `math.ceil(total_count / page_size)`
- `linear_trend_analysis()` — OLS regression on raw counts
- `log_trend_analysis()` — Exponential fit via log-transformation on non-zero years; returns annual growth rate and doubling time
- `compare_linear_vs_exponential()` — Selects best-fit model by R²
- `display_chart_control()` — Sets chart height by type and renders via `st.plotly_chart(use_container_width=True)`; no column wrapping, so charts naturally fill their container

**Module-level constants:**
- `URBAN_KEYWORDS` — the six tracked keywords; single definition used throughout the app

**Database credentials** are read from `.env` (local) or Streamlit secrets (deployed). The `.env` file contains `DB_URL`, `DB_KEY`, and `DB_CONNECTION_STRING` for Supabase.

**Styling** is applied via `assets/styles.css` (Montserrat font, custom colors for tags/sliders/multiselect, dark theme tweaks).

**Keywords tracked:** urban ecology, urban biodiversity, urban ecosystem, urban green spaces, urban vegetation, urban wildlife. Papers can belong to multiple keywords (stored comma-separated in `search_keyword`); the app expands these for per-keyword aggregation.

**Authors table:** exists in the database but is never queried by the app. It was a one-time enrichment pipeline in the notebook — used to resolve institution country codes via the ROR API and write them back into `papers.firstAuthorCountryIso`. The app reads that already-enriched column directly from the Papers table.
