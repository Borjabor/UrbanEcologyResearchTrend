"""
Urban Ecology Research Trends - Interactive Dashboard

This Streamlit app provides an interactive interface to explore urban ecology research trends based on data from OpenAlex API. Users can customize keyword selections, year ranges, and 
other parameters to generate dynamic visualizations.

Author: Your Name
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import pycountry
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def get_supabase_client():
    """
    Initialize Supabase client using environment variables or Streamlit secrets.
    """
    try:
        url = None
        key = None
        
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
    
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()


supabase: Client = get_supabase_client()

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

def get_all_papers_paginated(columns, filters=None, page_size=5000, show_progress=True, progress_placeholder=None):
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
            
            query = supabase.table('papers').select(columns).range(start, end)
            
            if filters:
                for filter_func in filters:
                    query = filter_func(query)
            
            result = query.execute()
            page_data = result.data
            
            if not page_data:
                break
                
            all_papers.extend(page_data)
            
            if show_progress:
                estimated_total = 66000
                current_progress = min(len(all_papers) / estimated_total, 1.0)
                progress_bar.progress(current_progress)
                status_text.text(f"Loaded {len(all_papers):,} papers...")
            
            if len(page_data) < page_size:
                break
                
            page += 1
            
            if page > 25:
                st.warning("Stopped pagination at 25 pages for safety")
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

@st.cache_data(ttl=3600, show_spinner=False)

def get_country_name(alpha3_code):
    """
    Convert alpha-3 country codes to full country names using pycountry.
    """
    try:
        country = pycountry.countries.get(alpha_3=alpha3_code)
        return country.name if country else alpha3_code
    except (KeyError, AttributeError):
        return alpha3_code

def load_data():
    """
    Load data from Supabase using pagination to get all records.
    This replicates the notebook's SQL query and data processing exactly.
    """
    
    loading_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with loading_placeholder:
        st.info("Data Loading Notice: This app loads 66,000+ research papers. Initial loading may take several seconds, but subsequent interactions will be instant thanks to caching.")
    
    with st.spinner("Loading data from Supabase... This may take a moment."):
        try:
            
            filters = [
                lambda q: q.not_.is_('search_keyword', 'null'),
                lambda q: q.neq('search_keyword', '')
            ]
            
            with status_placeholder:
                st.info("Loading papers data (this may take a few seconds)...")
            
            all_papers_data = get_all_papers_paginated(
                'paperId, year, search_keyword, title, authors, firstAuthorCountryIso, citationCount',
                filters=filters,
                page_size=5000,
                show_progress=True,
                progress_placeholder=progress_placeholder
            )
            
            df_all = pd.DataFrame(all_papers_data)
            
            df_raw = df_all.groupby(['year', 'search_keyword']).size().reset_index()
            df_raw.columns = ['year', 'search_keyword', 'paper_count']
        
            rows = []
            total_rows = []
            
            for _, row in df_raw.iterrows():
                keywords = [k.strip() for k in row['search_keyword'].split(',')]
                
                total_rows.append({
                    'year': row['year'], 
                    'search_keyword': 'total', 
                    'paper_count': row['paper_count']
                })
                
                for keyword in keywords:
                    if keyword:
                        rows.append({
                            'year': row['year'], 
                            'search_keyword': keyword, 
                            'paper_count': row['paper_count']
                        })
            
            df_keywords = pd.DataFrame(rows).groupby(['year', 'search_keyword'])['paper_count'].sum().reset_index()
            df_totals = pd.DataFrame(total_rows).groupby(['year', 'search_keyword'])['paper_count'].sum().reset_index()
            
            df_totals['search_keyword'] = df_totals['search_keyword'].replace('total', 'Total (All Keywords)')
            
            df_keywords_combined = pd.concat([df_keywords, df_totals], ignore_index=True)
            
            with status_placeholder:
                st.info("Loading detailed paper data...")
            
            original_keywords = [
                'urban ecology', 'urban biodiversity', 'urban ecosystem',
                'urban green spaces', 'urban vegetation', 'urban wildlife'
            ]
            
            expanded_data = []
            for _, row in df_all.iterrows():
                keywords_in_paper = [kw.strip() for kw in row['search_keyword'].split(',')]
                for keyword in keywords_in_paper:
                    if keyword in original_keywords:
                        expanded_data.append({
                            'paperId': row['paperId'],
                            'year': row['year'],
                            'search_keyword': keyword,
                            'title': row['title'],
                            'authors': row['authors'],
                            'firstAuthorCountryIso': row['firstAuthorCountryIso'],
                            'citationCount': row['citationCount']
                        })
            
            df_expanded = pd.DataFrame(expanded_data)
            
            df_papers_with_countries = df_all[
                (df_all['firstAuthorCountryIso'].notna()) & 
                (df_all['firstAuthorCountryIso'] != '')
            ]
            
            # df_countries = df_papers_with_countries.groupby('firstAuthorCountryIso').size().reset_index()
            # df_countries.columns = ['alpha3_code', 'paper_count']
            # df_countries = df_countries.sort_values('paper_count', ascending=False)
            
            
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
            
            return df_keywords_combined, df_countries, df_expanded, df_country_year_counts, total_unique_papers, df_totals
            
        except Exception as e:
            st.error(f"Error loading data from Supabase: {e}")
            return None, None, None, None, 0, None

def get_actual_keywords(selected_keywords):
    """
    Convert display keywords to actual keywords used in the DataFrame.
    """
    actual_keywords = []
    for keyword in selected_keywords:
        if keyword == "Total (All Keywords)":
            actual_keywords.append("Total (All Keywords)")
        else:
            actual_keywords.append(keyword)
    return actual_keywords


def linear_trend_analysis(df):
    """
    Perform linear regression analysis for each keyword in the dataframe
    Returns dataframe with comprehensive statistics for each keyword.
    """
    results = []
    
    for keyword in df['search_keyword'].unique():
        keyword_data = df[df['search_keyword'] == keyword].sort_values('year')
        
        if len(keyword_data) < 3:
            continue
        
        years = keyword_data['year'].values
        paper_counts = keyword_data['paper_count'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, paper_counts)
        
        r_squared = r_value**2
        
        n = len(years)
        t_val = stats.t.ppf(0.975, n-2)
        
        s_yx = np.sqrt(np.sum((paper_counts - (slope * years + intercept))**2) / (n - 2))
        s_xx = np.sum((years - np.mean(years))**2)
        slope_se = s_yx / np.sqrt(s_xx)
        
        slope_ci_lower = slope - t_val * slope_se
        slope_ci_upper = slope + t_val * slope_se
        
        is_significant = p_value < 0.05
        
        if is_significant:
            if slope > 0:
                trend_direction = "Increasing"
                interpretation = f"Significant upward trend: +{slope:.1f} papers/year"
            else:
                trend_direction = "Decreasing"
                interpretation = f"Significant downward trend: {slope:.1f} papers/year"
        else:
            trend_direction = "No significant trend"
            interpretation = "No statistically significant trend detected"
        
        results.append({
            'keyword': keyword,
            'slope': slope,
            'slope_ci_lower': slope_ci_lower,
            'slope_ci_upper': slope_ci_upper,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_error': std_err,
            'significant_trend': is_significant,
            'trend_direction': trend_direction,
            'interpretation': interpretation,
            'total_papers': paper_counts.sum(),
            'years_analyzed': n,
            'avg_papers_per_year': paper_counts.mean()
        })
    
    return pd.DataFrame(results)


def log_trend_analysis(df, min_papers=1):
    """
    Perform logarithmic regression analysis for each keyword to test exponential growth
    Returns dataframe with exponential growth statistics.
    """
    results = []
    
    for keyword in df['search_keyword'].unique():
        keyword_data = df[df['search_keyword'] == keyword].sort_values('year')
        
        if keyword_data['paper_count'].sum() < min_papers or len(keyword_data) < 3:
            continue
        
        years = keyword_data['year'].values
        paper_counts = keyword_data['paper_count'].values
        
        log_counts = np.log(paper_counts + 1)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_counts)
        
        r_squared = r_value**2
        
        annual_growth_rate = (np.exp(slope) - 1) * 100
        
        if slope > 0:
            doubling_time = np.log(2) / slope
        else:
            doubling_time = np.inf
        
        is_significant = p_value < 0.05
        
        if is_significant:
            if annual_growth_rate > 0:
                interpretation = f"Exponential growth: {annual_growth_rate:.1f}% per year"
            else:
                interpretation = f"Exponential decline: {annual_growth_rate:.1f}% per year"
        else:
            interpretation = "No significant exponential trend"
        
        results.append({
            'keyword': keyword,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'annual_growth_rate_percent': annual_growth_rate,
            'doubling_time_years': doubling_time,
            'significant_trend': is_significant,
            'interpretation': interpretation,
            'total_papers': paper_counts.sum(),
            'years_analyzed': len(years)
        })
    
    return pd.DataFrame(results)


def compare_linear_vs_exponential(df, min_papers=1):
    """
    Compare linear vs exponential growth models for each keyword
    Returns comparison dataframe with model selection results.
    """
    
    linear_results = linear_trend_analysis(df)
    log_results = log_trend_analysis(df, min_papers)
    
    if linear_results.empty and log_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    comparison = pd.merge(linear_results[['keyword', 'r_squared', 'p_value', 'significant_trend']], 
                         log_results[['keyword', 'r_squared', 'p_value', 'significant_trend', 'annual_growth_rate_percent', 'doubling_time_years']], 
                         on='keyword', 
                         suffixes=('_linear', '_log'))
    
    comparison['better_fit'] = np.where(comparison['r_squared_log'] > comparison['r_squared_linear'], 
                                       'Exponential', 'Linear')
    comparison['r_squared_difference'] = comparison['r_squared_log'] - comparison['r_squared_linear']
    
    comparison['model_recommendation'] = comparison.apply(lambda row: 
        f"Use {row['better_fit']} model (ΔR² = {row['r_squared_difference']:.3f})", axis=1)
    
    return comparison, linear_results, log_results

def display_chart_control(fig, chart_type="default"):
    """
    Display chart with responsive width control using native Streamlit columns.
    Uses very small margins that become negligible on mobile devices.
    """
    BASE_HEIGHT = 800
    
    if chart_type in ["map", "choropleth"]:
        col_ratios = [0.05, 0.90, 0.05]
        final_height = int(BASE_HEIGHT * 0.8)
    elif chart_type in ["time_series"]:
        col_ratios = [0.05, 0.90, 0.05]
        final_height = int(BASE_HEIGHT)
    elif chart_type in ["top_countries"]:
        col_ratios = [0.05, 0.90, 0.05]
        final_height = int(BASE_HEIGHT * 0.65)
    else:
        col_ratios = [0.075, 0.85, 0.075]
        final_height = BASE_HEIGHT
    
    col1, col2, col3 = st.columns(col_ratios)
    
    fig.update_layout(
        height=final_height,
        autosize=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    with col2:
        st.plotly_chart(
            fig, 
            use_container_width=True,
            height=final_height,
            config={'responsive': False, 'displayModeBar': False}
        )


# ==========================================
# MAIN APP
# ==========================================

def main():
    """
    Main function that creates the Streamlit app interface and handles all interactions.
    """
    
    st.title("Urban Ecology Research Trends")
    
    st.markdown("""
    **Interactive Dashboard for Urban Ecology Research Analysis**
    
    This dashboard allows you to explore trends in urban ecology research publications 
    from 1970-2023. Customize your analysis by selecting keywords, adjusting year ranges, 
    and modifying various parameters.
    """)
    
    df_keywords, df_countries, df_expanded, df_country_year_counts, total_unique_papers, df_totals = load_data()
    
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
        max_value=2023,
        value=(1970, 2023),
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
        
        df_country_totals = (
            df_country_year_counts.groupby(['country', 'alpha3_code'])['paper_count']
            .sum()
            .reset_index()
            .rename(columns={'paper_count': 'total_count'})
        )
        
        df_countries_filtered = (
            df_countries
            .groupby(['country', 'alpha3_code', 'year'])
            .agg(paper_count=('paperId', 'nunique'))
            .reset_index()
        )
        
        
        df_top_country_totals = df_country_totals.copy()
        df_top_country_totals = df_top_country_totals.sort_values('total_count', ascending=False).head(top_n_countries)
        
        top_countries = df_top_country_totals['country'].tolist()
        df_counts_filtered = df_country_year_counts[df_country_year_counts['country'].isin(top_countries)]
        
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
            title="Global Distribution of Urban Ecology Research",
            labels={'total_count': 'Number of Papers'},
            range_color=[0, df_country_totals['total_count'].max()]
        )
        
        fig_choropleth.update_layout(
            template='plotly_dark',
            autosize=False,
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
                height=map_height,
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
                            title=country_name,
                            labels={'year': 'Year', 'paper_count': 'Papers'}
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            autosize=True,
                            height=200,
                            margin=dict(l=10, r=10, t=20, b=0),
                            title_font_size=13,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"country_chart_{country_code}")
                    else:
                        st.write(f"*No data for {country_name}*")
          
        st.write('_' * 60)
        st.subheader("Top Research-Producing Countries")
            
        top_country_height = 500
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Country Research Output")
            
            # Root node
            labels = ["All Papers"]
            ids = ["root"]
            parents = [""]
            values = [df_top_country_totals['total_count'].sum()]

            # Country nodes
            for _, row in df_top_country_totals.iterrows():
                labels.append(f"{row['country']}<br>({row['total_count']:,})")
                ids.append(f"country_{row['country']}")
                parents.append("root")
                values.append(row['total_count'])
                
            colors = [np.nan] + values[1:]
                
            fig_treemap = go.Figure(go.Treemap(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(
                    colors=colors,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Number of Papers",
                        x=1.05,
                        len=0.8
                    )
                ),
                maxdepth=2,
                hoverinfo="skip",
                root_color="lightgrey",
                branchvalues="total"
            ))

            fig_treemap.update_layout(
                title="Total Research Output by Country",
                margin=dict(l=0, r=0, t=40, b=0),
                autosize=True,
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )

            st.plotly_chart(fig_treemap, use_container_width=True, height=top_country_height, key="treemap_main")
            
        with chart_col2:
            st.markdown("#### Output For Each Keyword")
            
            original_keywords = [
                'urban ecology', 'urban biodiversity', 'urban ecosystem',
                'urban green spaces', 'urban vegetation', 'urban wildlife'
            ]
            
            df_countries_exploded = df_countries.copy()
            
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
                df_counts_keywords[df_counts_keywords['country'].isin(top_countries) & df_counts_keywords['keywords'].isin(original_keywords)]
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
                title='Urban Ecology Research Output: Top Countries vs Keywords',
                margin=dict(l=0, r=0, t=40, b=0),
                autosize=True,
                xaxis_title='Keyword',
                yaxis_title='Country',
                template='plotly_dark',
                font=dict(size=12)
            )
            
                
            st.plotly_chart(fig_keyword_output_heatmap, use_container_width=True, height=top_country_height)
    
                
        
        st.subheader("Geographic Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Countries Analyzed", len(df_countries_filtered))
        
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
            title=f"Research Publications Over Time ({min_year}-{max_year})",
            labels={
                'year': 'Year',
                'paper_count': 'Number of Papers',
                'search_keyword': 'Keyword'
            }
        )
        
        fig_time.update_layout(
            template='plotly_dark',
            autosize=True,
            hovermode='x unified',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        display_chart_control(fig_time, "time_series")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Total (All Keywords)" in selected_keywords:
                df_totals_filtered = df_totals[
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                total_papers = df_totals_filtered['paper_count'].sum()
            else:
                years_with_keywords = df_filtered['year'].unique()
                df_totals_for_years = df_totals[
                    (df_totals['year'].isin(years_with_keywords)) &
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                total_papers = df_totals_for_years['paper_count'].sum()
            
            st.metric("Total Papers", f"{total_papers:,}")
        
        with col2:
            if "Total (All Keywords)" in selected_keywords:
                df_totals_filtered = df_totals[
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                avg_per_year = df_totals_filtered['paper_count'].mean()
            else:
                years_with_keywords = df_filtered['year'].unique()
                df_totals_for_years = df_totals[
                    (df_totals['year'].isin(years_with_keywords)) &
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                avg_per_year = df_totals_for_years['paper_count'].mean()
            
            st.metric("Avg Papers/Year", f"{avg_per_year:.1f}")
        
        with col3:
            years_span = max_year - min_year + 1
            st.metric("Years Analyzed", years_span)
    
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
        - **Exponential Model**: Assumes percentage growth (papers grow by a percentage each year)
        - **R² Score**: Measures how well the model fits the data (1.0 = perfect fit)
        - **Better Model**: The model with higher R² score fits the data better
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
            display_comparison['Better Model'] = display_comparison['better_fit']
            
            display_cols = ['keyword', 'Linear R²', 'Exponential R²', 'Better Model', 
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
                    title="Model Fit Comparison (R² values)",
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
                            linear_rates.append(linear_data['slope'].iloc[0] if not linear_data.empty else 0)
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
                        title="Growth Rates by Best-Fit Model<br><sub>Bar color according to best fit model</sub>",
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
                            title="Exponential Growth Rate (%/year)",
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
                    linear_pred = linear_kw['slope'] * year_range_extended + linear_kw['intercept']
                    log_pred = np.exp(log_kw['slope'] * year_range_extended + log_kw['intercept']) - 1
                    
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
                        title=f"Growth Models Comparison: {selected_keyword_viz}",
                        xaxis_title="Year",
                        yaxis_title="Number of Papers",
                        template='plotly_dark',
                        autosize=True,
                        showlegend=True,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    display_chart_control(fig_models, "time_series")
                    
                    better_model = "Exponential" if log_kw['r_squared'] > linear_kw['r_squared'] else "Linear"
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if better_model == "Linear":
                            st.markdown("**Growth Model: Linear**")
                            st.write(f"• Growth rate: {linear_kw['slope']:.2f} papers/year")
                            st.write(f"• R² score: {linear_kw['r_squared']:.3f}")
                            st.write(f"• P-value: {linear_kw['p_value']:.2e}")
                            st.write(f"• Significant: {'Yes' if linear_kw['significant_trend'] else 'No'}")
                        else:
                            st.markdown("**Growth Model: Exponential**")
                            st.write(f"• Growth rate: {log_kw['annual_growth_rate_percent']:.1f}%/year")
                            st.write(f"• Doubling time: {log_kw['doubling_time_years']:.1f} years")
                            st.write(f"• R² score: {log_kw['r_squared']:.3f}")
                            st.write(f"• P-value: {log_kw['p_value']:.2e}")
                            st.write(f"• Significant: {'Yes' if log_kw['significant_trend'] else 'No'}")
                    
                    with col2:
                        r_squared_diff = abs(log_kw['r_squared'] - linear_kw['r_squared'])
                        st.markdown("**Model Comparison:**")
                        st.write(f"• Better model: {better_model}")
                        st.write(f"• R² difference: {r_squared_diff:.3f}")
                        if r_squared_diff < 0.05:
                            st.write("• Very similar fit")
                        elif r_squared_diff < 0.1:
                            st.write("• Moderately better fit")
                        else:
                            st.write("• Much better fit")
            
        else:
            st.warning("Not enough data points for regression analysis with current selections.")
    
    

if __name__ == "__main__":
    main()
