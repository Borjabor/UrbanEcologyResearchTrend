# Project Feedback: Urban Ecology Research Trends

## Positives

**Statistical rigor.** The regression analysis is well-implemented — linear regression includes slope confidence intervals, R², p-values, and standard error. The exponential model correctly uses log-transformation and converts the slope back to an interpretable annual growth rate and doubling time. The model comparison via R² is a clean approach for a domain-appropriate question.

**Caching strategy.** Using `@st.cache_data(ttl=2592000)` for the 66K-paper dataset and a shorter TTL for country name lookups is sensible and makes interactions fast after the first load. The pagination loop to bypass Supabase's row limit is a practical solution.

**Visual coherence.** The dark theme is applied consistently across both Plotly charts and the custom CSS. The color palette for linear/exponential models is consistent across every chart that uses them. The custom CSS is polished — sliding sidebar animation, styled multiselect tags with hover effects, custom slider knobs — giving the app a finished feel beyond a typical Streamlit default.

**Good UX details.** Tooltips on sidebar controls, the loading notice explaining why the first load is slow, unified hover mode on the time series chart, responsive column layout, and the selectbox to drill into a specific keyword in the regression tab all show care for the user experience.

**Credential handling.** The env-var → Streamlit secrets fallback is correct for dual local/deployed use, with a meaningful error message when neither is found.

**Clear project documentation.** The README includes not just schema documentation but also actual research findings and interpretations, making the project readable as a research output rather than just a code repo.

---

## Issues & Improvement Suggestions

### Bugs

~~**Geographic tab ignores keyword and year filters.**~~
~~The choropleth map, treemap, and country×keyword heatmap are built from `df_country_totals` and `df_country_year_counts`, which are computed from all keywords and all years in `load_data()` before any sidebar filter is applied. Only the inline per-country time series mini-charts (inside the scrollable container) respect the year range. The effect is that switching keywords in the sidebar visually changes nothing on the geographic tab except those mini-charts, which is inconsistent and confusing.~~
✅ Fixed (2026-02-19): All geographic charts now derive from a `df_countries_geo` base that is pre-filtered by the sidebar year range and keyword selection before any aggregation. The country×keyword heatmap also scopes its columns to the active keyword selection.

### Code Quality

**`original_keywords` is defined in two places.**
Lines 238–241 and 772–775 both define the same list. This should be a single module-level constant (e.g., `URBAN_KEYWORDS = [...]`) to avoid the lists drifting out of sync.

~~**`supabase` client is instantiated at module level before `st.set_page_config`.**~~
~~Streamlit's docs require `set_page_config` to be the first Streamlit call. The module-level `supabase = get_supabase_client()` runs first and may call `st.error()` / `st.stop()` before page config is set, which will raise a Streamlit error. The client should be initialized inside `main()` or lazy-initialized the first time it's needed.~~
✅ Fixed (2026-02-19): `get_supabase_client()` is now decorated with `@st.cache_resource` (Streamlit's recommended pattern for singleton connections) and the module-level call removed. The client is instantiated lazily on first query, after `set_page_config` has run.

**`load_data()` has UI side-effects inside a cached function.**
`@st.cache_data` caches the return value, but `load_data()` also creates progress bars and `st.empty()` placeholders. On a cache hit the function body doesn't run at all, so there are no orphaned UI elements. On a cache miss, the placeholders are explicitly cleared at the end of the function, which mitigates the multi-user overlap concern. The main residual issue is that the function is harder to test or reuse outside a Streamlit context.

**`display_chart_control` is called inside an existing column context.**
The regression tab creates a two-column layout (`col1, col2`), then calls `display_chart_control` inside each column, which itself calls `st.columns`. Note: Streamlit added nested column support in v1.31 (January 2024), so this may work correctly on recent versions. If the deployed environment is on an older version, this would silently produce broken layouts.

**Hardcoded magic values.**
- `estimated_total = 66000` (line 132): hardcoded estimate used for the progress bar. Should at least be a named constant.
- Year range `(1970, 2023)` is hardcoded in the slider (line 544). Ideally this would be derived from the actual data min/max year.
- `page > 25` safety limit (line 142) is also a hardcoded guard with no documentation of why 25.

**Commented-out code (lines 527–528).**
The "Clear Cache" button is commented out. Either restore it (it's a useful debug tool) or remove the comment.

### Statistical / Analytical

**`log(y + 1)` transformation introduces bias for small counts.**
`log_trend_analysis()` adds 1 before taking the log (line 389) to avoid `log(0)`. For years with very few papers (e.g., 1970–1980), this +1 offset meaningfully distorts the regression slope upward. A cleaner approach is to exclude years with zero papers from the log regression, or to use a weighted regression that down-weights sparse early years.

**Keyword double-counting is not disclosed to users.**
Papers with multiple keywords (e.g., tagged `urban ecology, urban biodiversity`) are counted once toward each keyword. This means selecting all keywords individually produces a higher total than "Total (All Keywords)". The app doesn't explain this, and the "Time Series Summary" total papers metric behaves differently depending on whether "Total" is in the selection (lines 880–913). A note in the UI would prevent misinterpretation.

### Missing Features / Gaps

**The notebook analyses are not in the app.**
The Jupyter notebook contains keyword co-occurrence/similarity analysis and the urban vs. general ecology comparison — arguably the most interesting findings in the README. None of these are in the Streamlit app. A "Keyword Relationships" tab would meaningfully extend the dashboard.

**No data freshness indicator.**
Data is cached for a month, but users have no way to know when it was last loaded. A simple "Data last refreshed: X" line using `st.session_state` or a timestamp stored alongside the cached data would address this.

**The Authors table is loaded but never used.**
The database schema includes an Authors table with institution and country data, but the app only queries the Papers table. The `firstAuthorId` foreign key and the richer geographic data from the Authors table (`country_name`, `last_known_institution_name`) go unused. Joining to Authors could fill in country data for papers where `firstAuthorCountryIso` is null, reducing geographic data gaps.
