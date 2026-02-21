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

## Architecture

This is a single-file Streamlit app (`streamlit_app.py`) that visualizes urban ecology research trends (1970–2023) using data from a Supabase (PostgreSQL) backend populated via the OpenAlex and ROR APIs.

**Data flow:**
1. `get_supabase_client()` initializes the Supabase connection once via `@st.cache_resource` (lazy, called on first query)
2. `load_data()` fetches 66,000+ papers from Supabase via pagination, cached for 1 month (`@st.cache_data(ttl=2592000)`)
3. Data is filtered by user-selected keywords and year range
4. Analysis functions compute regression models and geographic counts
5. Plotly renders interactive charts in three tabs

**Three UI tabs:**
- **Geographic Analysis** — Choropleth map, country time series, treemap, country×keyword heatmap. All charts are filtered by the sidebar year range, keyword selection, and top-N country count. A `df_countries_geo` base is derived from the raw `df_countries` at the top of the tab before any aggregation.
- **Time Series** — Publication counts over time by keyword
- **Regression Analysis** — Linear vs. exponential growth model comparison with R² scores

**Key analysis functions in `streamlit_app.py`:**
- `get_supabase_client()` — `@st.cache_resource` singleton; returns the Supabase client, creating it only once
- `linear_trend_analysis()` — OLS regression on raw counts
- `log_trend_analysis()` — Exponential fit via log-transformation; returns annual growth rate and doubling time
- `compare_linear_vs_exponential()` — Selects best-fit model by R²
- `display_chart_control()` — Wraps Plotly charts in responsive Streamlit columns

**Database credentials** are read from `.env` (local) or Streamlit secrets (deployed). The `.env` file contains `DB_URL`, `DB_KEY`, and `DB_CONNECTION_STRING` for Supabase.

**Styling** is applied via `assets/styles.css` (Montserrat font, custom colors for tags/sliders/multiselect, dark theme tweaks).

**Keywords tracked:** urban ecology, urban biodiversity, urban ecosystem, urban green spaces, urban vegetation, urban wildlife. Papers can belong to multiple keywords (stored comma-separated in `search_keyword`); the app expands these for per-keyword aggregation.
