"""
Urban Ecology Research Trends - Interactive Dashboard

This Streamlit app provides an interactive interface to explore urban ecology research trends based on data from OpenAlex API. Users can customize keyword selections, year ranges, and 
other parameters to generate dynamic visualizations.

Author: Andre Borja Miranda
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.optimize import curve_fit
import pycountry
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

URBAN_KEYWORDS = [
                'urban ecology', 'urban biodiversity', 'urban ecosystem',
                'urban green spaces', 'urban vegetation', 'urban wildlife'
            ]

@st.cache_resource
def get_supabase_client() -> Client:
    """
    Initialize and cache the Supabase client (one instance per session).
    Credentials come from environment variables (local) with a fallback to
    Streamlit secrets (deployed).
    """
    url = os.getenv("DB_URL")
    key = os.getenv("DB_KEY")

    if not url or not key:
        try:
            if hasattr(st, 'secrets') and 'DB_URL' in st.secrets:
                url = st.secrets["DB_URL"]
                key = st.secrets["DB_KEY"]
        except Exception:
            pass

    if not url or not key:
        st.error("Database credentials not found. Please check your .env file or Streamlit secrets.")
        st.stop()

    return create_client(url, key)


st.set_page_config(
    page_title="Urban Ecology Research Trends",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="auto"
)


def load_css():
    """Load custom CSS from external file"""
    css_files = ['assets/styles.css', 'styles.css']
    
    for css_file in css_files:
        try:
            with open(css_file, 'r') as f:
                css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            return
        except FileNotFoundError:
            continue
    
    # Fallback inline styles if no CSS file found
    st.markdown("""
    <style>
        .css-1d391kg {
            width: 350px !important;
        }
        section[data-testid="stSidebar"] {
            width: 350px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 350px !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_all_papers_paginated(columns, filters=None, page_size=5000, show_progress=True, progress_placeholder=None, total_count=None):
    """Get all papers using optimized pagination to bypass max rows limit."""
    all_papers = []
    page = 0

    if show_progress:
        if progress_placeholder:
            progress_bar = progress_placeholder.progress(0)
            status_text = progress_placeholder.empty()
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
        status_text.text("Loading data...")

    while True:
        try:
            start = page * page_size
            end = start + page_size - 1

            query = get_supabase_client().table('papers').select(columns).range(start, end)

            if filters:
                for filter_func in filters:
                    query = filter_func(query)

            result = query.execute()
            page_data = result.data

            if not page_data:
                break

            all_papers.extend(page_data)

            if show_progress and total_count:
                current_progress = min(len(all_papers) / total_count, 1.0)
                progress_bar.progress(current_progress)
                status_text.text(f"Loaded {len(all_papers):,} of {total_count:,} papers...")

            if len(page_data) < page_size:
                break

            page += 1

            # Derive max pages from total_count when available; the fallback
            # value is arbitrary — no documented limit exists in Supabase/PostgREST.
            max_pages = math.ceil(total_count / page_size) if total_count else 50
            if page >= max_pages:
                st.warning("Reached maximum page limit — some records may be missing.")
                break

        except Exception as e:
            st.error(f"Error on page {page + 1}: {e}")
            break
    
    if show_progress:
        if progress_placeholder:
            progress_placeholder.empty()
        else:
            progress_bar.empty()
            status_text.empty()
    
    return all_papers

def get_country_name(alpha3_code):
    """
    Convert an alpha-3 country code to a full country name using pycountry.
    Not cached: this is an in-memory O(1) lookup, so Streamlit's cache machinery
    would cost more than it saves. pycountry.countries.get returns None (it does
    not raise) for unknown codes, so we just fall back to the code itself.
    """
    country = pycountry.countries.get(alpha_3=alpha3_code)
    return country.name if country else alpha3_code

@st.cache_data(ttl=2592000) #One month data caching
def load_data():
    """
    Load data from Supabase using pagination to get all records.
    This replicates the notebook's SQL query and data processing exactly.
    """
    
    loading_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with loading_placeholder:
        st.info("Data Loading Notice: This app loads 140,000+ research papers. Initial loading may take several seconds, but subsequent interactions will be instant thanks to caching.")
    
    with st.spinner("Loading data from Supabase... This may take a moment."):
        try:
            
            filters = [
                lambda q: q.not_.is_('search_keyword', 'null'),
                lambda q: q.neq('search_keyword', '')
            ]

            count_result = (
                get_supabase_client()
                .table('papers')
                .select('*', count='exact', head=True)
                .not_.is_('search_keyword', 'null')
                .neq('search_keyword', '')
                .execute()
            )
            total_count = count_result.count

            with status_placeholder:
                st.info(f"Loading {total_count:,} papers (this may take a few seconds)...")

            all_papers_data = get_all_papers_paginated(
                'paperId, year, search_keyword, firstAuthorCountryIso',
                filters=filters,
                total_count=total_count,
                page_size=5000,
                show_progress=True,
                progress_placeholder=progress_placeholder
            )

            df_all = pd.DataFrame(all_papers_data)

            df_raw = df_all.groupby(['year', 'search_keyword']).size().reset_index()
            df_raw.columns = ['year', 'search_keyword', 'paper_count']

            # Expand comma-separated keywords (vectorized) so each keyword gets its
            # own per-year count; multi-keyword rows contribute to each keyword.
            df_keywords = df_raw.copy()
            df_keywords['search_keyword'] = df_keywords['search_keyword'].str.split(',')
            df_keywords = df_keywords.explode('search_keyword')
            df_keywords['search_keyword'] = df_keywords['search_keyword'].str.strip()
            df_keywords = df_keywords[df_keywords['search_keyword'] != '']
            df_keywords = (
                df_keywords.groupby(['year', 'search_keyword'])['paper_count']
                .sum()
                .reset_index()
            )

            # Count unique papers per year directly so multi-keyword papers are
            # never double-counted in the Total line regardless of how
            # search_keyword is stored in the database.
            df_totals = (
                df_all.groupby('year')
                .agg(paper_count=('paperId', 'nunique'))
                .reset_index()
            )
            df_totals['search_keyword'] = 'Total (All Keywords)'
            
            df_keywords_combined = pd.concat([df_keywords, df_totals], ignore_index=True)
            
            df_countries = (
                df_all[
                    df_all['firstAuthorCountryIso'].notnull() & 
                    (df_all['firstAuthorCountryIso'] != '')]
                [['paperId', 'year', 'search_keyword', 'firstAuthorCountryIso']]
                .rename(columns={'firstAuthorCountryIso': 'alpha3_code'})
                .sort_values('year')
                .reset_index(drop=True)
            )
            
            df_countries['country'] =df_countries['alpha3_code'].apply(get_country_name)
            
            df_country_year_counts = (
                df_countries
                .groupby(['year', 'country', 'alpha3_code', 'search_keyword'])
                .agg(paper_count=('paperId', 'nunique'))
                .reset_index()
            )
            
            
            loading_placeholder.empty()
            status_placeholder.empty()
            progress_placeholder.empty()
            
            total_unique_papers = len(df_all)
            
            return df_keywords_combined, df_countries, df_country_year_counts, total_unique_papers, df_totals
            
        except Exception as e:
            st.error(f"Error loading data from Supabase: {e}")
            return None, None, None, 0, None

def linear_trend_analysis(df):
    """
    Linear (OLS) regression on raw paper counts for each keyword.
    R-squared here is in count space, directly comparable to the exponential model.
    """
    results = []

    for keyword in df['search_keyword'].unique():
        keyword_data = df[df['search_keyword'] == keyword].sort_values('year')
        years = keyword_data['year'].values.astype(float)
        counts = keyword_data['paper_count'].values.astype(float)

        if len(years) < 3:
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
        r_squared = r_value ** 2

        n = len(years)
        t_val = stats.t.ppf(0.975, n - 2)
        # std_err from linregress IS the slope standard error -- no need to recompute.
        slope_ci_lower = slope - t_val * std_err
        slope_ci_upper = slope + t_val * std_err

        results.append({
            'keyword': keyword,
            'slope_papers_per_year': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_error': std_err,
            'slope_ci_lower': slope_ci_lower,
            'slope_ci_upper': slope_ci_upper,
            'significant_trend': p_value < 0.05,
            'total_papers': counts.sum(),
            'avg_papers_per_year': counts.mean(),
            'years_analyzed': n,
        })

    return pd.DataFrame(results)


def _exp_model(t, a, b):
    """Exponential growth model fit in count space: y = a * exp(b * t)."""
    return a * np.exp(b * t)


def log_trend_analysis(df, min_papers=1):
    """
    Exponential growth fit per keyword, estimated by NONLINEAR least squares in
    count space (y = a * exp(b * (year - year0))). This replaces the old straight-
    line-on-log(y) fit, which minimises relative error and once back-transformed
    overshoots recent high-count years -- making its count-space fit look worse
    than it is and spuriously flipping the model choice. R-squared is reported in
    count space so it is comparable to the linear model. Equivalent log-linear
    coefficients (log_slope, log_intercept) are returned so plotting code that
    draws exp(log_slope*year + log_intercept) reproduces the curve.
    """
    results = []

    for keyword in df['search_keyword'].unique():
        keyword_data = df[df['search_keyword'] == keyword].sort_values('year')
        years = keyword_data['year'].values.astype(float)
        counts = keyword_data['paper_count'].values.astype(float)

        if len(years) < 3 or counts.sum() < min_papers:
            continue

        year0 = years.min()
        t = years - year0

        try:
            popt, pcov = curve_fit(
                _exp_model, t, counts,
                p0=[max(counts[0], 1.0), 0.05],
                maxfev=20000
            )
        except (RuntimeError, ValueError):
            continue

        a, b = popt
        predicted = _exp_model(t, a, b)

        ss_res = np.sum((counts - predicted) ** 2)
        ss_tot = np.sum((counts - counts.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        n = len(years)
        b_std_err = np.sqrt(pcov[1, 1]) if np.all(np.isfinite(pcov)) else np.nan
        if b_std_err and np.isfinite(b_std_err) and b_std_err > 0:
            t_stat = b / b_std_err
            p_value = 2 * stats.t.sf(np.abs(t_stat), n - 2)
            t_val = stats.t.ppf(0.975, n - 2)
            slope_ci_lower = b - t_val * b_std_err
            slope_ci_upper = b + t_val * b_std_err
        else:
            p_value = np.nan
            slope_ci_lower = slope_ci_upper = np.nan

        annual_growth_rate = (np.exp(b) - 1) * 100
        doubling_time = np.log(2) / b if b > 0 else np.inf

        # Equivalent log-linear coefficients (see docstring).
        log_slope = b
        log_intercept = np.log(a) - b * year0

        results.append({
            'keyword': keyword,
            'exp_a': a,
            'exp_b': b,
            'year0': year0,
            'log_slope': log_slope,
            'log_intercept': log_intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_error': b_std_err,
            'slope_ci_lower': slope_ci_lower,
            'slope_ci_upper': slope_ci_upper,
            'annual_growth_rate_percent': annual_growth_rate,
            'doubling_time_years': doubling_time,
            'significant_trend': (not np.isnan(p_value)) and p_value < 0.05,
            'total_papers': counts.sum(),
            'avg_papers_per_year': counts.mean(),
            'years_analyzed': n,
        })

    return pd.DataFrame(results)


def _aic_count_space(rss, n, k=3):
    """Gaussian-error AICc from a count-space residual sum of squares. k = slope +
    intercept + error variance = 3 for both models, so the base penalty is
    identical; AICc adds the small-sample correction 2k(k+1)/(n-k-1), which
    matters at n in the low tens."""
    aic = n * np.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        aic += (2 * k * (k + 1)) / (n - k - 1)
    return aic


def compare_linear_vs_exponential(df, min_papers=1):
    """
    Compare linear vs exponential growth per keyword. Both models are scored in
    COUNT space (papers/year) and the winner is chosen by AICc, so the two are
    judged on the same ruler. |delta-AICc| < 2 means the models are statistically
    indistinguishable (Burnham & Anderson) -- reported as a tie rather than a
    winner crowned on noise. The old code compared a count-space R2 (linear)
    against a log-space R2 (exponential), which is invalid and biased toward
    'Exponential'.
    """
    linear_results = linear_trend_analysis(df)
    log_results = log_trend_analysis(df, min_papers)

    if linear_results.empty or log_results.empty:
        return pd.DataFrame(), linear_results, log_results

    rows = []
    for keyword in linear_results['keyword']:
        lin_match = linear_results[linear_results['keyword'] == keyword]
        exp_match = log_results[log_results['keyword'] == keyword]
        if lin_match.empty or exp_match.empty:
            continue
        lin = lin_match.iloc[0]
        exp = exp_match.iloc[0]

        keyword_data = df[df['search_keyword'] == keyword].sort_values('year')
        years = keyword_data['year'].values.astype(float)
        counts = keyword_data['paper_count'].values.astype(float)
        n = len(years)

        pred_lin = lin['slope_papers_per_year'] * years + lin['intercept']
        pred_exp = np.exp(exp['log_slope'] * years + exp['log_intercept'])

        rss_lin = np.sum((counts - pred_lin) ** 2)
        rss_exp = np.sum((counts - pred_exp) ** 2)

        aic_lin = _aic_count_space(rss_lin, n)
        aic_exp = _aic_count_space(rss_exp, n)

        # |delta-AICc| < 2 -> statistically indistinguishable; report a tie.
        if abs(aic_lin - aic_exp) < 2:
            better_fit = 'Indistinguishable'
        else:
            better_fit = 'Exponential' if aic_exp < aic_lin else 'Linear'

        rows.append({
            'keyword': keyword,
            'r_squared_linear': lin['r_squared'],
            'r_squared_log': exp['r_squared'],            # count-space R2 of the exp fit
            'r_squared_difference': exp['r_squared'] - lin['r_squared'],
            'aic_linear': aic_lin,
            'aic_log': aic_exp,
            'delta_aic': aic_lin - aic_exp,               # > 0 favours exponential
            'p_value_linear': lin['p_value'],
            'p_value_log': exp['p_value'],
            'significant_trend_linear': lin['significant_trend'],
            'significant_trend_log': exp['significant_trend'],
            'annual_growth_rate_percent': exp['annual_growth_rate_percent'],
            'doubling_time_years': exp['doubling_time_years'],
            'better_fit': better_fit,
        })

    return pd.DataFrame(rows), linear_results, log_results

def display_chart_control(fig, chart_type="default"):
    """
    Display chart with height scaling by type. Uses use_container_width=True
    so the chart fills its natural container — full page at top level, or the
    enclosing column width when called inside a st.columns context.
    """
    BASE_HEIGHT = 800

    heights = {
        "map":          int(BASE_HEIGHT * 0.8),
        "choropleth":   int(BASE_HEIGHT * 0.8),
        "time_series":  BASE_HEIGHT,
        "top_countries": int(BASE_HEIGHT * 0.65),
    }
    final_height = heights.get(chart_type, BASE_HEIGHT)

    fig.update_layout(
        height=final_height,
        autosize=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False}
    )


# ==========================================
# MAIN APP
# ==========================================

def main():
    """
    Main function that creates the Streamlit app interface and handles all interactions.
    """
    
    st.title("Urban Ecology Research Trends")
    
    st.header("Interactive Dashboard for Urban Ecology Research Analysis")
    
    st.subheader("This dashboard allows you to explore trends in urban ecology research publications from 1970-2025. Customize your analysis by selecting keywords, and adjusting year ranges.")
    
    df_keywords, df_countries, df_country_year_counts, total_unique_papers, df_totals = load_data()
    
    if df_keywords is None:
        st.error("Failed to load data from Supabase. Please check the connection and table setup.")
        st.info("Common issues:")
        st.write("- Row Level Security (RLS) may be blocking access")
        st.write("- Table may be empty or have different column names")
        st.write("- API key may not have sufficient permissions")
        return
    
    if 'show_loading_messages' not in st.session_state:
        st.session_state.show_loading_messages = True
    
    
    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.header("Analysis Controls")
    st.sidebar.text("(Note: you can collapse this sidebar with the arrow at the top)")

    available_keywords = sorted(df_keywords['search_keyword'].unique())
    keyword_options = ["Total (All Keywords)"] + available_keywords
    
    st.sidebar.subheader("Select Keywords")
    selected_keywords = st.sidebar.multiselect(
        "Choose keywords to analyze:",
        keyword_options,
        default=available_keywords,
        help="Select one or more keywords to include in the analysis. 'Total' shows aggregated data across all keywords."
    )
    
    st.sidebar.subheader("Year Range")
    min_year, max_year = st.sidebar.slider(
        "Select year range:",
        min_value=1970,
        max_value=2025,
        value=(1970, 2025),
        help="Adjust the time period for analysis"
    )
    
    st.sidebar.subheader("Country Selection")
    top_n_countries = st.sidebar.slider(
        "Top N countries to display:",
        min_value=5,
        max_value=30,
        value=20,
        help="Number of top countries to show in geographical visualizations"
    )
    
    if not selected_keywords:
        st.warning("Please select at least one keyword to begin analysis.")
        return
    
    if "Total (All Keywords)" in selected_keywords:
        df_totals_filtered = df_totals[
            (df_totals['year'] >= min_year) & 
            (df_totals['year'] <= max_year)
        ].copy()
        df_totals_filtered['search_keyword'] = 'Total (All Keywords)'
        
        if len(selected_keywords) == 1:
            df_filtered = df_totals_filtered
        else:
            individual_keywords = [kw for kw in selected_keywords if kw != "Total (All Keywords)"]
            df_individual = df_keywords[
                (df_keywords['search_keyword'].isin(individual_keywords)) &
                (df_keywords['year'] >= min_year) &
                (df_keywords['year'] <= max_year)
            ]
            df_filtered = pd.concat([df_individual, df_totals_filtered], ignore_index=True)
    else:
        df_filtered = df_keywords[
            (df_keywords['search_keyword'].isin(selected_keywords)) &
            (df_keywords['year'] >= min_year) &
            (df_keywords['year'] <= max_year)
        ]
    
    # ==========================================
    # MAIN CONTENT TABS
    # ==========================================
    
    tab1, tab2, tab3 = st.tabs([
        "Geographic Analysis",
        "Time Series", 
        "Regression Analysis",
    ])
    
    # ==========================================
    # TAB 1: GEOGRAPHIC ANALYSIS
    # ==========================================
    with tab1:
        st.header("World Research Distribution")
            
        left_col, right_col = st.columns([4, 1])
        
        # Apply sidebar filters to geographic data
        active_geo_keywords = [kw for kw in selected_keywords if kw != "Total (All Keywords)"]
        use_all_keywords = "Total (All Keywords)" in selected_keywords

        df_countries_geo = df_countries[
            (df_countries['year'] >= min_year) &
            (df_countries['year'] <= max_year)
        ].copy()

        if not use_all_keywords and active_geo_keywords:
            mask = df_countries_geo['search_keyword'].apply(
                lambda x: any(kw in [k.strip() for k in x.split(',')] for kw in active_geo_keywords)
            )
            df_countries_geo = df_countries_geo[mask]

        df_country_totals = (
            df_countries_geo
            .groupby(['country', 'alpha3_code'])
            .agg(total_count=('paperId', 'nunique'))
            .reset_index()
        )

        df_countries_filtered = (
            df_countries_geo
            .groupby(['country', 'alpha3_code', 'year'])
            .agg(paper_count=('paperId', 'nunique'))
            .reset_index()
        )

        df_top_country_totals = df_country_totals.sort_values('total_count', ascending=False).head(top_n_countries)
        top_countries = df_top_country_totals['country'].tolist()

        df_countries_filtered = df_countries_filtered[df_countries_filtered['country'].isin(top_countries)]
        df_countries_filtered['country'] = pd.Categorical(df_countries_filtered['country'], categories=top_countries, ordered=True)
        df_countries_filtered = df_countries_filtered.sort_values(['country', 'year']).reset_index(drop=True)
        
        map_height = 600
        
        fig_choropleth = px.choropleth(
            df_country_totals,
            locations='alpha3_code',
            color='total_count',
            hover_name='country',
            hover_data={'alpha3_code': False, 'total_count': ':,'},
            color_continuous_scale='Viridis',
            labels={'total_count': 'Number of Papers'},
            range_color=[0, df_country_totals['total_count'].max()]
        )
        
        fig_choropleth.update_layout(
            template='plotly_dark',
            autosize=False,
            height=map_height,
            title=dict(
                text="Global Distribution of Urban Ecology Research",
                font=dict(
                    family="Montserrat light, Helvetica, Arial, sans-serif"
                )
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                center=dict(lat=10, lon=0),
            ),
            margin=dict(l=0, r=0, t=25, b=0)
        )
        
        with left_col:
            st.plotly_chart(
                fig_choropleth,
                use_container_width=True,
                config={'responsive': False, 'displayModeBar': False}
            )
            
        with right_col:
            st.markdown(f'**Top {top_n_countries} Countries Time Series**')
            
            
            with st.container(height=map_height):
                for country, group in df_countries_filtered.groupby('country', sort=False):
                    country_code = group['alpha3_code'].iloc[0]
                    country_name = country

                    country_ts = group[
                        (group['year'] >= min_year) &
                        (group['year'] <= max_year)
                    ]
        
                    if not country_ts.empty:
                        fig = px.line(
                            country_ts,
                            x='year',
                            y='paper_count',
                            labels={'year': 'Year', 'paper_count': 'Papers'}
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            autosize=True,
                            title=dict(
                                text=country_name,
                                font=dict(
                                    family="Montserrat light, Helvetica, Arial, sans-serif"
                                )
                            ),
                            height=200,
                            margin=dict(l=10, r=10, t=20, b=0),
                            title_font_size=13,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"country_chart_{country_code}")
                    else:
                        st.write(f"*No data for {country_name}*")
          

        st.write('_' * 60)
        st.header("Top Research-Producing Countries")
            
        top_country_height = 500
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Country Research Output")

            # px.treemap builds the root + country nodes and branch totals from the
            # dataframe directly (replaces the hand-assembled ids/labels/parents lists).
            fig_treemap = px.treemap(
                df_top_country_totals,
                path=[px.Constant('All Papers'), 'country'],
                values='total_count',
                color='total_count',
                color_continuous_scale='Viridis',
                labels={'total_count': 'Number of Papers'},
            )
            fig_treemap.update_traces(
                texttemplate='%{label}<br>%{value:,}',
                hovertemplate='<b>%{label}</b><br>Papers: %{value:,}<extra></extra>',
            )
            fig_treemap.update_layout(
                title=dict(
                    text="Total Research Output by Country",
                    font=dict(
                        family="Montserrat light, Helvetica, Arial, sans-serif"
                    )
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                autosize=True,
                height=top_country_height,
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                coloraxis_colorbar=dict(title="Number of Papers")
            )

            st.plotly_chart(fig_treemap, use_container_width=True, key="treemap_main")
            
        with chart_col2:
            st.subheader("Output For Each Keyword")
            
            heatmap_keywords = active_geo_keywords if (active_geo_keywords and not use_all_keywords) else URBAN_KEYWORDS

            df_countries_exploded = df_countries_geo.copy()

            df_countries_exploded['keywords'] = df_countries_exploded['search_keyword'].str.split(',')
            df_countries_exploded = df_countries_exploded.explode('keywords')
            df_countries_exploded['keywords'] = df_countries_exploded['keywords'].str.strip()

            df_counts_keywords = (
                df_countries_exploded
                .groupby(['country', 'keywords', 'alpha3_code'])
                .agg(paper_count=('paperId', 'nunique'))
                .reset_index()
            )

            heatmap_data = (
                df_counts_keywords[df_counts_keywords['country'].isin(top_countries) & df_counts_keywords['keywords'].isin(heatmap_keywords)]
                .pivot(index='country', columns='keywords', values='paper_count')
                .fillna(0)
            )

            fig_keyword_output_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                colorbar=dict(title='Number of Papers'),
                hovertemplate='<b>%{y}</b><br>Keyword: %{x}<br>Papers: %{z}<extra></extra>',
                showscale=True
            ))

            fig_keyword_output_heatmap.update_layout(
                title=dict(
                    text='Urban Ecology Research Output: Top Countries vs Keywords',
                    font=dict(
                        family="Montserrat light, Helvetica, Arial, sans-serif"
                    )
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                autosize=True,
                xaxis_title='Keyword',
                yaxis_title='Country',
                template='plotly_dark',
                height=top_country_height,
                font=dict(size=12)
            )

            st.plotly_chart(fig_keyword_output_heatmap, use_container_width=True)
    
                
        st.write('_' * 60)
        st.subheader("Geographic Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Countries Analyzed", df_countries_filtered['country'].nunique())
        
        with col2:
            st.metric("Leading Country", df_countries_filtered.iloc[0]['country'])
        
        with col3:
            total_papers_geo = df_countries_filtered['paper_count'].sum()
            st.metric("Total Papers (Top Countries)", f"{total_papers_geo:,}")
        
            
    # ==========================================
    # TAB 2: TIME SERIES ANALYSIS
    # ==========================================
    with tab2:
        st.header("Time Series Analysis")
        st.markdown("Explore how research volume has changed over time for different keywords.")
        
        fig_time = px.line(
            df_filtered, 
            x='year', 
            y='paper_count', 
            color='search_keyword',
            labels={
                'year': 'Year',
                'paper_count': 'Number of Papers',
                'search_keyword': 'Keyword'
            }
        )
        
        fig_time.update_layout(
            template='plotly_dark',
            autosize=True,
            title=dict(
                text=f"Research Publications Over Time ({min_year}-{max_year})",
                font=dict(
                    family="Montserrat light, Helvetica, Arial, sans-serif"
                )
            ),
            hovermode='x unified',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        display_chart_control(fig_time, "time_series")
        
        st.write('_' * 60)
        st.subheader("Time Series Summary")

        # Filter the totals once, then read sum/mean off it (the per-column blocks
        # used to re-derive this identical slice four times).
        ts_summary = df_totals[(df_totals['year'] >= min_year) & (df_totals['year'] <= max_year)]
        if "Total (All Keywords)" not in selected_keywords:
            years_with_keywords = df_filtered['year'].unique()
            ts_summary = ts_summary[ts_summary['year'].isin(years_with_keywords)]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", f"{ts_summary['paper_count'].sum():,}")
        with col2:
            st.metric("Avg Papers/Year", f"{ts_summary['paper_count'].mean():.1f}")
        with col3:
            st.metric("Years Analyzed", max_year - min_year + 1)
    
    # ==========================================
    # TAB 3: REGRESSION ANALYSIS
    # ==========================================
    with tab3:
        st.header("Growth Trend Analysis")
        st.markdown("Statistical analysis of research growth trends - Linear vs Exponential models.")
        
        LINEAR_COLOR = "#4ECDAF"
        EXPONENTIAL_COLOR = "#C541CA"
        DATA_POINTS_COLOR = '#F39C12'
        
        st.markdown("""
        **Understanding the models:**
        - **Linear Model**: Assumes constant growth (same number of papers added each year)
        - **Exponential Model**: Assumes percentage growth (papers grow by a percentage each year), fit by nonlinear least squares in count space
        - **R² Score**: How well each model fits the data (1.0 = perfect fit); both are measured in count space so they are comparable
        - **Better Model**: Chosen by AICc (lower is better). **ΔAICc > 0 favours exponential**, and **|ΔAICc| < 2** means the two models are statistically indistinguishable
        """)
        
        with st.spinner("Computing regression statistics..."):
            comparison, linear_results, log_results = compare_linear_vs_exponential(df_filtered)
        
        if not comparison.empty:
            st.subheader("Model Comparison: Linear vs Exponential")
            
            display_comparison = comparison.copy()
            display_comparison['Annual Growth Rate (%)'] = display_comparison['annual_growth_rate_percent'].round(2)
            display_comparison['Doubling Time (years)'] = display_comparison['doubling_time_years'].round(1)
            display_comparison['Linear R²'] = display_comparison['r_squared_linear'].round(3)
            display_comparison['Exponential R²'] = display_comparison['r_squared_log'].round(3)
            display_comparison['ΔAICc'] = display_comparison['delta_aic'].round(1)
            display_comparison['Better Model'] = display_comparison['better_fit']

            display_cols = ['keyword', 'Linear R²', 'Exponential R²', 'ΔAICc', 'Better Model',
                          'Annual Growth Rate (%)', 'Doubling Time (years)']
            st.dataframe(
                display_comparison[display_cols].rename(columns={'keyword': 'Keyword'}),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rsquared = go.Figure()
                
                fig_rsquared.add_trace(go.Bar(
                    name='Linear Model',
                    x=comparison['keyword'],
                    y=comparison['r_squared_linear'],
                    marker_color=LINEAR_COLOR,
                    opacity=0.7
                ))
                
                fig_rsquared.add_trace(go.Bar(
                    name='Exponential Model',
                    x=comparison['keyword'],
                    y=comparison['r_squared_log'],
                    marker_color=EXPONENTIAL_COLOR,
                    opacity=0.7
                ))
                
                fig_rsquared.update_layout(
                    title=dict(
                        text="Model Fit Comparison (R² values)",
                        font=dict(
                            family="Montserrat light, Helvetica, Arial, sans-serif"
                        )
                    ),
                    xaxis_title="Keywords",
                    yaxis_title="R² Score",
                    template='plotly_dark',
                    autosize=True,
                    barmode='group',
                    showlegend=True,
                    legend=dict(
                        x=0.4,
                        y=1.22,
                        bgcolor="rgba(0,0,0,0.5)"
                    ),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                display_chart_control(fig_rsquared, "default")
            
            with col2:
                if not comparison.empty:
                    keywords = comparison['keyword'].tolist()
                    better_models = comparison['better_fit'].tolist()
                    
                    linear_rates = []
                    exponential_rates = []
                    
                    for keyword in keywords:
                        linear_data = linear_results[linear_results['keyword'] == keyword]
                        log_data = log_results[log_results['keyword'] == keyword]
                        better_model = comparison[comparison['keyword'] == keyword]['better_fit'].iloc[0]
                        
                        if better_model == "Linear":
                            linear_rates.append(linear_data['slope_papers_per_year'].iloc[0] if not linear_data.empty else 0)
                            exponential_rates.append(None)
                        else:
                            linear_rates.append(None)
                            exp_rate = log_data['annual_growth_rate_percent'].iloc[0] if not log_data.empty else 0
                            
                            if -100 < exp_rate < 100:
                                exponential_rates.append(exp_rate)
                            else:
                                exponential_rates.append(None)
                    
                    fig_growth = go.Figure()
                    
                    fig_growth.add_trace(go.Bar(
                        name='Linear Growth (papers/year)',
                        x=keywords,
                        y=linear_rates,
                        marker_color=LINEAR_COLOR,
                        opacity=0.8,
                        yaxis='y1'
                    ))
                    
                    fig_growth.add_trace(go.Bar(
                        name='Exponential Growth (%/year)',
                        x=keywords,
                        y=exponential_rates,
                        marker_color=EXPONENTIAL_COLOR,
                        opacity=0.8,
                        yaxis='y2'
                    ))
                    
                    fig_growth.update_layout(
                        title=dict(
                            text="Growth Rates by Best-Fit Model<br><sub>Bar color according to best fit model</sub>",
                            font=dict(
                                family="Montserrat light, Helvetica, Arial, sans-serif"
                            )
                        ),
                        xaxis_title="Keywords",
                        template='plotly_dark',
                        autosize=True,
                        yaxis=dict(
                            title="Linear Growth Rate (papers/year)",
                            title_font=dict(color=LINEAR_COLOR),
                            tickfont=dict(color=LINEAR_COLOR),
                            side="left",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(92, 232, 237, 0.3)',
                            griddash='dot'
                        ),
                        yaxis2=dict(
                            title=dict(
                                text="Exponential Growth Rate (%/year)",
                                font=dict(
                                    family="Montserrat light, Helvetica, Arial, sans-serif"
                                )
                            ),
                            title_font=dict(color=EXPONENTIAL_COLOR),
                            tickfont=dict(color=EXPONENTIAL_COLOR),
                            overlaying="y",
                            side="right",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(237, 92, 229, 0.3)',
                        ),
                        showlegend=True,
                        legend=dict(
                            x=0.4,
                            y=1.22,
                            bgcolor="rgba(0,0,0,0.5)"
                        ),
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    display_chart_control(fig_growth, "default")
            
            st.subheader("Growth Model Visualizations")
            
            available_viz_keywords = sorted(df_filtered['search_keyword'].unique())
            
            selected_keyword_viz = st.selectbox(
                "Select keyword to visualize growth models:",
                available_viz_keywords,
                help="Choose a keyword to see linear vs exponential model fits"
            )
            
            if selected_keyword_viz:
                keyword_data = df_filtered[df_filtered['search_keyword'] == selected_keyword_viz].sort_values('year')
                
                if len(keyword_data) >= 3:
                    years = keyword_data['year'].values
                    counts = keyword_data['paper_count'].values
                    
                    linear_kw = linear_results[linear_results['keyword'] == selected_keyword_viz].iloc[0]
                    log_kw = log_results[log_results['keyword'] == selected_keyword_viz].iloc[0]
                    
                    year_range_extended = np.linspace(years.min(), years.max(), 100)
                    linear_pred = linear_kw['slope_papers_per_year'] * year_range_extended + linear_kw['intercept']
                    log_pred = np.exp(log_kw['log_slope'] * year_range_extended + log_kw['log_intercept'])
                    
                    fig_models = go.Figure()
                    
                    fig_models.add_trace(go.Scatter(
                        x=year_range_extended,
                        y=linear_pred,
                        mode='lines',
                        name=f'Linear (R²={linear_kw["r_squared"]:.3f})',
                        line=dict(color=LINEAR_COLOR, width=2)
                    ))
                    
                    fig_models.add_trace(go.Scatter(
                        x=year_range_extended,
                        y=log_pred,
                        mode='lines',
                        name=f'Exponential (R²={log_kw["r_squared"]:.3f})',
                        line=dict(color=EXPONENTIAL_COLOR, width=2)
                    ))
                    
                    fig_models.add_trace(go.Scatter(
                        x=years,
                        y=counts,
                        mode='markers',
                        name='Actual Data',
                        marker=dict(size=10, color=DATA_POINTS_COLOR, symbol='circle')
                    ))
                    
                    fig_models.update_layout(
                        title=dict(
                            text=f"Growth Models Comparison: {selected_keyword_viz}",
                            font=dict(
                                family="Montserrat light, Helvetica, Arial, sans-serif"
                            )
                        ),
                        xaxis_title="Year",
                        yaxis_title="Number of Papers",
                        template='plotly_dark',
                        autosize=True,
                        showlegend=True,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    display_chart_control(fig_models, "time_series")
                    
                    # Model choice comes from the AICc comparison (count space), not
                    # a cross-scale R² contest. May be a tie ('Indistinguishable').
                    comp_row = comparison[comparison['keyword'] == selected_keyword_viz].iloc[0]
                    better_model = comp_row['better_fit']
                    delta_aic = comp_row['delta_aic']

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        if better_model == "Linear":
                            st.markdown("**Growth Model: Linear**")
                            st.write(f"• Growth rate: {linear_kw['slope_papers_per_year']:.2f} papers/year")
                            st.write(f"• R² score: {linear_kw['r_squared']:.3f}")
                            st.write(f"• P-value: {linear_kw['p_value']:.2e}")
                            st.write(f"• Significant: {'Yes' if linear_kw['significant_trend'] else 'No'}")
                        elif better_model == "Exponential":
                            st.markdown("**Growth Model: Exponential**")
                            st.write(f"• Growth rate: {log_kw['annual_growth_rate_percent']:.1f}%/year")
                            st.write(f"• Doubling time: {log_kw['doubling_time_years']:.1f} years")
                            st.write(f"• R² score: {log_kw['r_squared']:.3f}")
                            st.write(f"• P-value: {log_kw['p_value']:.2e}")
                            st.write(f"• Significant: {'Yes' if log_kw['significant_trend'] else 'No'}")
                        else:
                            st.markdown("**Growth Model: Indistinguishable**")
                            st.write(f"• Linear: {linear_kw['slope_papers_per_year']:.2f} papers/year (R²={linear_kw['r_squared']:.3f})")
                            st.write(f"• Exponential: {log_kw['annual_growth_rate_percent']:.1f}%/year, doubling {log_kw['doubling_time_years']:.1f} yr (R²={log_kw['r_squared']:.3f})")
                            st.write("• Both fits are statistically equivalent here")

                    with col2:
                        st.markdown("**Model Comparison:**")
                        st.write(f"• Better model: {better_model}")
                        st.write(f"• ΔAICc: {delta_aic:+.1f}")
                        st.caption("ΔAICc > 0 favours exponential; |ΔAICc| < 2 is a statistical tie.")
            
        else:
            st.warning("Not enough data points for regression analysis with current selections.")
    
    

if __name__ == "__main__":
    main()
