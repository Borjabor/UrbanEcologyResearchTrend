"""
Urban Ecology Research Trends - Interactive Dashboard

This Streamlit app provides an interactive interface to explore urban ecology research trends
based on data from OpenAlex API. Users can customize keyword selections, year ranges, and 
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

# Load environment variables for local development
load_dotenv()

# ==========================================
# SUPABASE CONNECTION SETUP
# ==========================================

def get_supabase_client():
    """
    Initialize Supabase client using environment variables or Streamlit secrets.
    """
    try:
        url = None
        key = None
        
        # Try environment variables first (for local development)
        url = os.getenv("DB_URL")
        key = os.getenv("DB_KEY")
        
        # If not found, try Streamlit secrets (for deployment)
        if not url or not key:
            try:
                if hasattr(st, 'secrets') and 'DB_URL' in st.secrets:
                    url = st.secrets["DB_URL"]
                    key = st.secrets["DB_KEY"]
            except Exception:
                pass  # Secrets not available, that's okay for local development
        
        if not url or not key:
            st.error("Database credentials not found. Please check your .env file or Streamlit secrets.")
            st.stop()
        
        return create_client(url, key)
    
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# Initialize Supabase client
supabase: Client = get_supabase_client()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Urban Ecology Research Trends",
    page_icon="🏙️",
    layout="wide",  # Use full width of the browser
    initial_sidebar_state="expanded"  # Keep sidebar open by default
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_all_papers_paginated(columns, filters=None, page_size=5000, show_progress=True, progress_placeholder=None):
    """Get all papers using optimized pagination to bypass max rows limit."""
    all_papers = []
    page = 0
    
    # Initialize progress bar if requested
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
            
            # Build query
            query = supabase.table('papers').select(columns).range(start, end)
            
            # Apply filters if provided
            if filters:
                for filter_func in filters:
                    query = filter_func(query)
            
            result = query.execute()
            page_data = result.data
            
            if not page_data:
                break
                
            all_papers.extend(page_data)
            
            # Update progress
            if show_progress:
                # Estimate progress (we know there are ~66k papers)
                estimated_total = 66000
                current_progress = min(len(all_papers) / estimated_total, 1.0)
                progress_bar.progress(current_progress)
                status_text.text(f"Loaded {len(all_papers):,} papers...")
            
            if len(page_data) < page_size:
                # Last page
                break
                
            page += 1
            
            # Safety break
            if page > 25:  # Reduced since we're using 5k page size (66k/5k = ~14 pages)
                st.warning("Stopped pagination at 25 pages for safety")
                break
                
        except Exception as e:
            st.error(f"Error on page {page + 1}: {e}")
            break
    
    # Clear progress indicators
    if show_progress:
        if progress_placeholder:
            progress_placeholder.empty()
        else:
            progress_bar.empty()
            status_text.empty()
    
    return all_papers

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour, custom spinner
def load_data():
    """
    Load data from Supabase using pagination to get all records.
    This replicates the notebook's SQL query and data processing exactly.
    """
    # Create placeholders for loading messages that we can clear later
    loading_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with loading_placeholder:
        st.info("Data Loading Notice: This app loads 66,000+ research papers. Initial loading may take 30-60 seconds, but subsequent interactions will be instant thanks to caching.")
    
    with st.spinner("Loading data from Supabase... This may take a moment."):
        try:
            # STEP 1: Load all papers with pagination
            # Equivalent to: SELECT year, search_keyword FROM papers 
            # WHERE search_keyword IS NOT NULL AND search_keyword != ''
            
            filters = [
                lambda q: q.not_.is_('search_keyword', 'null'),
                lambda q: q.neq('search_keyword', '')
            ]
            
            with status_placeholder:
                st.info("Loading papers data (this may take a few seconds)...")
            
            all_papers_data = get_all_papers_paginated(
                'paperId, year, search_keyword',  # Include paperId for unique counting
                filters=filters,
                page_size=5000,  # Larger page size for fewer requests
                show_progress=True,
                progress_placeholder=progress_placeholder
            )
            
            df_all = pd.DataFrame(all_papers_data)
            
            # Replicate the exact SQL aggregation
            df_raw = df_all.groupby(['year', 'search_keyword']).size().reset_index()
            df_raw.columns = ['year', 'search_keyword', 'paper_count']
        
            # STEP 2: Apply EXACT same expansion logic as notebook
            rows = []
            total_rows = []
            
            for _, row in df_raw.iterrows():
                keywords = [k.strip() for k in row['search_keyword'].split(',')]
                
                # Total tracking (exactly like notebook)
                total_rows.append({
                    'year': row['year'], 
                    'search_keyword': 'total', 
                    'paper_count': row['paper_count']
                })
                
                # Keyword expansion (exactly like notebook)
                for keyword in keywords:
                    if keyword:
                        rows.append({
                            'year': row['year'], 
                            'search_keyword': keyword, 
                            'paper_count': row['paper_count']  # Use full aggregated count
                        })
            
            # STEP 3: Final aggregation (exactly like notebook)
            df_keywords = pd.DataFrame(rows).groupby(['year', 'search_keyword'])['paper_count'].sum().reset_index()
            df_totals = pd.DataFrame(total_rows).groupby(['year', 'search_keyword'])['paper_count'].sum().reset_index()
            
            # Rename 'total' to match app expectations
            df_totals['search_keyword'] = df_totals['search_keyword'].replace('total', 'Total (All Keywords)')
            
            # Combine for app use
            df_keywords_combined = pd.concat([df_keywords, df_totals], ignore_index=True)
            
            # STEP 4: Create expanded individual paper data for other analyses
            with status_placeholder:
                st.info("Loading detailed paper data...")
            
            all_papers_full_data = get_all_papers_paginated(
                'paperId, year, search_keyword, title, authors, firstAuthorCountryIso, citationCount',
                filters=filters,
                page_size=5000,
                show_progress=False  # Don't show progress for subsequent calls
            )
            
            df_all_papers = pd.DataFrame(all_papers_full_data)
            
            # Define the 6 original keywords
            original_keywords = [
                'urban ecology', 'urban biodiversity', 'urban ecosystem',
                'urban green spaces', 'urban vegetation', 'urban wildlife'
            ]
            
            # Create expanded dataset for other analyses
            expanded_data = []
            for _, row in df_all_papers.iterrows():
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
            
            # Optimize: Reuse the same full dataset for country data instead of separate queries
            df_papers_with_countries = df_all_papers[
                (df_all_papers['firstAuthorCountryIso'].notna()) & 
                (df_all_papers['firstAuthorCountryIso'] != '')
            ]
            
            # Country data from existing dataset
            df_countries = df_papers_with_countries.groupby('firstAuthorCountryIso').size().reset_index()
            df_countries.columns = ['alpha3_code', 'paper_count']
            df_countries = df_countries.sort_values('paper_count', ascending=False)
            
            # Country-year data from existing dataset
            df_country_years = df_papers_with_countries.groupby(['year', 'firstAuthorCountryIso']).size().reset_index()
            df_country_years.columns = ['year', 'country', 'paper_count']
            df_country_years = df_country_years.sort_values(['year', 'country'])
            
            # Clear all loading messages
            loading_placeholder.empty()
            status_placeholder.empty()
            progress_placeholder.empty()
            
            # IMPORTANT: Calculate total unique papers for accurate reporting
            total_unique_papers = len(df_all_papers)
            
            return df_keywords_combined, df_countries, df_expanded, df_country_years, total_unique_papers, df_totals
            
        except Exception as e:
            st.error(f"Error loading data from Supabase: {e}")
            return None, None, None, None, 0, None

def get_country_name(alpha3_code):
    """
    Convert alpha-3 country codes to full country names using pycountry.
    """
    try:
        country = pycountry.countries.get(alpha_3=alpha3_code)
        return country.name if country else alpha3_code
    except (KeyError, AttributeError):
        return alpha3_code


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
        
        if len(keyword_data) < 3:  # Need at least 3 data points
            continue
        
        years = keyword_data['year'].values
        paper_counts = keyword_data['paper_count'].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, paper_counts)
        
        r_squared = r_value**2
        
        # Calculate confidence intervals (95%)
        n = len(years)
        t_val = stats.t.ppf(0.975, n-2)  # 95% confidence interval
        
        # Standard error for slope
        s_yx = np.sqrt(np.sum((paper_counts - (slope * years + intercept))**2) / (n - 2))
        s_xx = np.sum((years - np.mean(years))**2)
        slope_se = s_yx / np.sqrt(s_xx)
        
        slope_ci_lower = slope - t_val * slope_se
        slope_ci_upper = slope + t_val * slope_se
        
        # Determine significance (p < 0.05)
        is_significant = p_value < 0.05
        
        # Determine trend direction and interpretation
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
        
        # Filter out keywords with very few papers
        if keyword_data['paper_count'].sum() < min_papers or len(keyword_data) < 3:
            continue
        
        years = keyword_data['year'].values
        paper_counts = keyword_data['paper_count'].values
        
        # Add small constant to handle zero counts for log transformation
        log_counts = np.log(paper_counts + 1)
        
        # Perform linear regression on log-transformed data
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_counts)
        
        r_squared = r_value**2
        
        # Calculate annual growth rate from log regression
        # slope in log space represents the growth rate
        annual_growth_rate = (np.exp(slope) - 1) * 100
        
        # Calculate doubling time (years)
        if slope > 0:
            doubling_time = np.log(2) / slope
        else:
            doubling_time = np.inf
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Interpretation
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
    # Get results from both analyses
    linear_results = linear_trend_analysis(df)
    log_results = log_trend_analysis(df, min_papers)
    
    if linear_results.empty and log_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Merge results for comparison
    comparison = pd.merge(linear_results[['keyword', 'r_squared', 'p_value', 'significant_trend']], 
                         log_results[['keyword', 'r_squared', 'p_value', 'significant_trend', 'annual_growth_rate_percent', 'doubling_time_years']], 
                         on='keyword', 
                         suffixes=('_linear', '_log'))
    
    # Determine which model fits better
    comparison['better_fit'] = np.where(comparison['r_squared_log'] > comparison['r_squared_linear'], 
                                       'Exponential', 'Linear')
    comparison['r_squared_difference'] = comparison['r_squared_log'] - comparison['r_squared_linear']
    
    # Add model selection interpretation
    comparison['model_recommendation'] = comparison.apply(lambda row: 
        f"Use {row['better_fit']} model (ΔR² = {row['r_squared_difference']:.3f})", axis=1)
    
    return comparison, linear_results, log_results

def create_keyword_similarity_matrix(df_expanded, selected_keywords, year_range):
    """
    Create keyword co-occurrence similarity matrix based on papers that contain
    multiple keywords. Shows which keywords appear together in the same papers.
    Note: Excludes 'Total' option as it represents aggregated data.
    """
    # Filter out the Total option for similarity analysis
    analysis_keywords = [kw for kw in selected_keywords if kw != "Total (All Keywords)"]
    
    if len(analysis_keywords) < 2:
        return None  # Need at least 2 keywords for similarity analysis
    
    # Filter by year range and selected keywords
    df_filtered = df_expanded[
        (df_expanded['year'] >= year_range[0]) & 
        (df_expanded['year'] <= year_range[1]) &
        (df_expanded['search_keyword'].isin(analysis_keywords))
    ]
    
    # Group by paper to see which keywords appear together
    paper_keywords = df_filtered.groupby('paperId')['search_keyword'].apply(list).reset_index()
    
    # Count co-occurrences
    cooccurrence_counts = {}
    keyword_counts = {}
    
    # Initialize counts
    for kw in analysis_keywords:
        keyword_counts[kw] = 0
        for kw2 in analysis_keywords:
            cooccurrence_counts[(kw, kw2)] = 0
    
    # Count occurrences and co-occurrences
    for _, row in paper_keywords.iterrows():
        keywords_in_paper = row['search_keyword']
        
        # Count individual keywords
        for kw in keywords_in_paper:
            if kw in analysis_keywords:
                keyword_counts[kw] += 1
        
        # Count co-occurrences (pairs)
        for i, kw1 in enumerate(keywords_in_paper):
            for kw2 in keywords_in_paper[i+1:]:
                if kw1 in analysis_keywords and kw2 in analysis_keywords:
                    cooccurrence_counts[(kw1, kw2)] += 1
                    cooccurrence_counts[(kw2, kw1)] += 1  # Symmetric
    
    # Create similarity matrix using Jaccard similarity
    similarity_matrix = pd.DataFrame(
        np.zeros((len(analysis_keywords), len(analysis_keywords))),
        index=analysis_keywords, 
        columns=analysis_keywords
    )
    
    for kw1 in analysis_keywords:
        for kw2 in analysis_keywords:
            if kw1 != kw2:
                # Jaccard similarity: intersection / union
                intersection = cooccurrence_counts[(kw1, kw2)]
                union = keyword_counts[kw1] + keyword_counts[kw2] - intersection
                if union > 0:
                    similarity = intersection / union
                    similarity_matrix.at[kw1, kw2] = similarity
            
    return similarity_matrix

# ==========================================
# MAIN APP
# ==========================================

def main():
    """
    Main function that creates the Streamlit app interface and handles all interactions.
    """
    
    # App title and description
    st.title("Urban Ecology Research Trends")
    
    st.markdown("""
    **Interactive Dashboard for Urban Ecology Research Analysis**
    
    This dashboard allows you to explore trends in urban ecology research publications 
    from 1970-2023. Customize your analysis by selecting keywords, adjusting year ranges, 
    and modifying various parameters.
    """)
    
    # Load data
    df_keywords, df_countries, df_expanded, df_country_years, total_unique_papers, df_totals = load_data()
    
    if df_keywords is None:
        st.error("Failed to load data from Supabase. Please check the connection and table setup.")
        st.info("Common issues:")
        st.write("- Row Level Security (RLS) may be blocking access")
        st.write("- Table may be empty or have different column names")
        st.write("- API key may not have sufficient permissions")
        return
    
    # Show data loaded confirmation with dismiss functionality
    if 'show_loading_messages' not in st.session_state:
        st.session_state.show_loading_messages = True
    
    if st.session_state.show_loading_messages:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.success(f"Data successfully loaded: {total_unique_papers:,} unique papers ready for analysis!")
            st.info("Tip: All interactions are now instant thanks to caching.")
        
        with col2:
            if st.button("✕ Dismiss", help="Hide these loading messages"):
                st.session_state.show_loading_messages = False
                st.rerun()
    
    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.header("Analysis Controls")
    
    # Get available keywords
    available_keywords = sorted(df_keywords['search_keyword'].unique())
    keyword_options = ["Total (All Keywords)"] + available_keywords
    
    # Keyword selection
    st.sidebar.subheader("Select Keywords")
    selected_keywords = st.sidebar.multiselect(
        "Choose keywords to analyze:",
        keyword_options,
        default=available_keywords,  # Select all individual keywords by default
        help="Select one or more keywords to include in the analysis. 'Total' shows aggregated data across all keywords."
    )
    
    # Year range selection
    st.sidebar.subheader("Year Range")
    min_year, max_year = st.sidebar.slider(
        "Select year range:",
        min_value=1970,
        max_value=2023,
        value=(1970, 2023),  # Default to full range
        help="Adjust the time period for analysis"
    )
    
    # Additional parameters
    st.sidebar.subheader("Display Options")
    
    # Number of countries to show in geographical analysis
    top_n_countries = st.sidebar.slider(
        "Top N countries to display:",
        min_value=5,
        max_value=30,
        value=20,
        help="Number of top countries to show in geographical visualizations"
    )
    
    # Chart theme
    chart_theme = st.sidebar.selectbox(
        "Chart theme:",
        ["plotly_dark", "plotly_white", "plotly"],
        index=0,
        help="Visual theme for all charts"
    )
    
    # Validate selections
    if not selected_keywords:
        st.warning("Please select at least one keyword to begin analysis.")
        return
    
    # Filter data based on selections
    # Handle "Total" option separately
    if "Total (All Keywords)" in selected_keywords:
        # Use the CORRECT df_totals for Total line, not the incorrectly calculated one
        df_totals_filtered = df_totals[
            (df_totals['year'] >= min_year) & 
            (df_totals['year'] <= max_year)
        ].copy()
        df_totals_filtered['search_keyword'] = 'Total (All Keywords)'  # Rename to match app expectations
        
        # If only Total is selected, use just the correct total data
        if len(selected_keywords) == 1:
            df_filtered = df_totals_filtered
        else:
            # Include both individual keywords and correct total
            individual_keywords = [kw for kw in selected_keywords if kw != "Total (All Keywords)"]
            df_individual = df_keywords[
                (df_keywords['search_keyword'].isin(individual_keywords)) &
                (df_keywords['year'] >= min_year) &
                (df_keywords['year'] <= max_year)
            ]
            df_filtered = pd.concat([df_individual, df_totals_filtered], ignore_index=True)
    else:
        # Regular filtering for individual keywords only
        df_filtered = df_keywords[
            (df_keywords['search_keyword'].isin(selected_keywords)) &
            (df_keywords['year'] >= min_year) &
            (df_keywords['year'] <= max_year)
        ]
    
    # ==========================================
    # MAIN CONTENT TABS
    # ==========================================
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Time Series", 
        "Keyword Relationships", 
        "Regression Analysis",
        "Geographic Analysis",
        "Data Summary"
    ])
    
    # ==========================================
    # TAB 1: TIME SERIES ANALYSIS
    # ==========================================
    with tab1:
        st.header("Time Series Analysis")
        st.markdown("Explore how research volume has changed over time for different keywords.")
        
        # Create time series plot
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
            template=chart_theme,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate total papers correctly using df_totals
            if "Total (All Keywords)" in selected_keywords:
                # Use the correct totals from df_totals filtered by year range
                df_totals_filtered = df_totals[
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                total_papers = df_totals_filtered['paper_count'].sum()
            else:
                # For individual keywords only, we need to calculate unique papers
                # Get the years that have the selected keywords
                years_with_keywords = df_filtered['year'].unique()
                df_totals_for_years = df_totals[
                    (df_totals['year'].isin(years_with_keywords)) &
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                total_papers = df_totals_for_years['paper_count'].sum()
            
            st.metric("Total Papers", f"{total_papers:,}")
        
        with col2:
            # Calculate average per year correctly using df_totals
            if "Total (All Keywords)" in selected_keywords:
                df_totals_filtered = df_totals[
                    (df_totals['year'] >= min_year) & 
                    (df_totals['year'] <= max_year)
                ]
                avg_per_year = df_totals_filtered['paper_count'].mean()
            else:
                # For individual keywords, use the same years as above
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
    # TAB 2: KEYWORD RELATIONSHIPS
    # ==========================================
    with tab2:
        st.header("Keyword Relationship Analysis")
        st.markdown("Analyze how often keywords appear together in research papers.")
        
        # Filter out total option for relationship analysis
        analysis_keywords = [kw for kw in selected_keywords if kw != "Total (All Keywords)"]
        
        if len(analysis_keywords) < 2:
            if "Total (All Keywords)" in selected_keywords:
                st.info("Keyword relationship analysis is not applicable when 'Total (All Keywords)' is selected, as it represents aggregated data. Please select at least 2 individual keywords to see their relationships.")
            else:
                st.warning("Please select at least 2 keywords to analyze relationships.")
        else:
            # Create similarity matrix
            with st.spinner("Computing keyword similarities..."):
                similarity_matrix = create_keyword_similarity_matrix(
                    df_expanded, selected_keywords, (min_year, max_year)
                )
            
            if similarity_matrix is not None:
                # Apply mask to hide upper triangle (since matrix is symmetric)
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                similarity_matrix_masked = similarity_matrix.copy()
                similarity_matrix_masked[mask] = np.nan
                
                # Create heatmap with masked data
                fig_heatmap = px.imshow(
                    similarity_matrix_masked,
                    title="Keyword Co-occurrence Similarity Matrix",
                    labels=dict(x="Keyword", y="Keyword", color="Similarity"),
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    zmin=0,  # Set minimum value for color scale
                    zmax=1   # Set maximum value for color scale
                )
            
            fig_heatmap.update_layout(
                template=chart_theme,
                height=600
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Explanation
            st.markdown("""
            **How to read this matrix:**
            - Values closer to 1 indicate keywords that frequently appear together in papers
            - Values closer to 0 indicate keywords that rarely appear together
            - Uses Jaccard similarity (intersection over union) to measure co-occurrence
            """)
            
            # Show top keyword pairs
            if st.checkbox("Show keyword co-occurrence pairs"):
                pairs_data = []
                for i, kw1 in enumerate(selected_keywords):
                    for j, kw2 in enumerate(selected_keywords):
                        if i < j:  # Avoid duplicates
                            similarity = similarity_matrix.loc[kw1, kw2]
                            pairs_data.append({
                                'Keyword 1': kw1,
                                'Keyword 2': kw2,
                                'Co-occurrence Score': similarity
                            })
                
                if pairs_data:
                    pairs_df = pd.DataFrame(pairs_data).sort_values('Co-occurrence Score', ascending=False)
                    st.dataframe(pairs_df, use_container_width=True)
    
    # ==========================================
    # ==========================================
    # TAB 3: REGRESSION ANALYSIS
    # ==========================================
    with tab3:
        st.header("Growth Trend Analysis")
        st.markdown("Statistical analysis of research growth trends - Linear vs Exponential models.")
        
        # Debug: Show data structure when Total is selected
        if st.checkbox("Show debug info for regression"):
            st.write("Debug Information:")
            st.write(f"Selected keywords: {selected_keywords}")
            st.write(f"Unique keywords in filtered data: {sorted(df_filtered['search_keyword'].unique())}")
            st.write(f"Data shape: {df_filtered.shape}")
            if not df_filtered.empty:
                st.write("Sample data:")
                st.write(df_filtered.head())
        
        # Perform regression analysis
        with st.spinner("Computing regression statistics..."):
            comparison, linear_results, log_results = compare_linear_vs_exponential(df_filtered)
        
        if not comparison.empty:
            # Display comparison results
            st.subheader("Model Comparison: Linear vs Exponential")
            
            # Format the results for display
            display_comparison = comparison.copy()
            display_comparison['Annual Growth Rate (%)'] = display_comparison['annual_growth_rate_percent'].round(2)
            display_comparison['Doubling Time (years)'] = display_comparison['doubling_time_years'].round(1)
            display_comparison['Linear R²'] = display_comparison['r_squared_linear'].round(3)
            display_comparison['Exponential R²'] = display_comparison['r_squared_log'].round(3)
            display_comparison['Better Model'] = display_comparison['better_fit']
            
            # Select columns for display
            display_cols = ['keyword', 'Linear R²', 'Exponential R²', 'Better Model', 
                          'Annual Growth Rate (%)', 'Doubling Time (years)']
            st.dataframe(
                display_comparison[display_cols].rename(columns={'keyword': 'Keyword'}),
                use_container_width=True
            )
            
            # Create visualization comparing models
            col1, col2 = st.columns(2)
            
            with col1:
                # R-squared comparison
                fig_rsquared = go.Figure()
                
                fig_rsquared.add_trace(go.Bar(
                    name='Linear Model',
                    x=comparison['keyword'],
                    y=comparison['r_squared_linear'],
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_rsquared.add_trace(go.Bar(
                    name='Exponential Model',
                    x=comparison['keyword'],
                    y=comparison['r_squared_log'],
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_rsquared.update_layout(
                    title="Model Fit Comparison (R² values)",
                    xaxis_title="Keywords",
                    yaxis_title="R² Score",
                    template=chart_theme,
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig_rsquared, use_container_width=True)
            
            with col2:
                # Growth rates for exponential model
                finite_growth = log_results[
                    (log_results['annual_growth_rate_percent'] > -100) & 
                    (log_results['annual_growth_rate_percent'] < 100)
                ]
                
                if not finite_growth.empty:
                    colors = ['green' if sig else 'red' for sig in finite_growth['significant_trend']]
                    
                    fig_growth = go.Figure(data=[
                        go.Bar(
                            x=finite_growth['keyword'],
                            y=finite_growth['annual_growth_rate_percent'],
                            marker_color=colors,
                            opacity=0.7
                        )
                    ])
                    
                    fig_growth.update_layout(
                        title="Annual Growth Rates (Exponential Model)<br><sub>Green=Significant, Red=Not Significant</sub>",
                        xaxis_title="Keywords",
                        yaxis_title="Annual Growth Rate (%)",
                        template=chart_theme,
                        height=400
                    )
                    
                    st.plotly_chart(fig_growth, use_container_width=True)
            
            # Model fit visualization for each keyword
            st.subheader("Growth Model Visualizations")
            
            # Get actual keywords available in the DataFrame after filtering
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
                    
                    # Get regression results for this keyword
                    linear_kw = linear_results[linear_results['keyword'] == selected_keyword_viz].iloc[0]
                    log_kw = log_results[log_results['keyword'] == selected_keyword_viz].iloc[0]
                    
                    # Generate predictions
                    year_range_extended = np.linspace(years.min(), years.max(), 100)
                    linear_pred = linear_kw['slope'] * year_range_extended + linear_kw['intercept']
                    log_pred = np.exp(log_kw['slope'] * year_range_extended + log_kw['intercept']) - 1
                    
                    # Create comparison plot
                    fig_models = go.Figure()
                    
                    # Linear model prediction line
                    fig_models.add_trace(go.Scatter(
                        x=year_range_extended,
                        y=linear_pred,
                        mode='lines',
                        name=f'Linear (R²={linear_kw["r_squared"]:.3f})',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Exponential model prediction line  
                    fig_models.add_trace(go.Scatter(
                        x=year_range_extended,
                        y=log_pred,
                        mode='lines',
                        name=f'Exponential (R²={log_kw["r_squared"]:.3f})',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Actual data points (on top so they're visible)
                    fig_models.add_trace(go.Scatter(
                        x=years,
                        y=counts,
                        mode='markers',
                        name='Actual Data',
                        marker=dict(size=10, color='gold', symbol='circle')
                    ))
                    
                    fig_models.update_layout(
                        title=f"Growth Models Comparison: {selected_keyword_viz}",
                        xaxis_title="Year",
                        yaxis_title="Number of Papers",
                        template=chart_theme,
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_models, use_container_width=True)
                    
                    # Show model statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Linear Model:**")
                        st.write(f"• Growth rate: {linear_kw['slope']:.2f} papers/year")
                        st.write(f"• R² score: {linear_kw['r_squared']:.3f}")
                        st.write(f"• P-value: {linear_kw['p_value']:.2e}")
                        st.write(f"• Significant: {'Yes' if linear_kw['significant_trend'] else 'No'}")
                    
                    with col2:
                        st.markdown("**Exponential Model:**")
                        st.write(f"• Growth rate: {log_kw['annual_growth_rate_percent']:.1f}%/year")
                        st.write(f"• Doubling time: {log_kw['doubling_time_years']:.1f} years")
                        st.write(f"• R² score: {log_kw['r_squared']:.3f}")
                        st.write(f"• P-value: {log_kw['p_value']:.2e}")
                        st.write(f"• Significant: {'Yes' if log_kw['significant_trend'] else 'No'}")
            
            # Explanation
            st.markdown("""
            **Understanding the models:**
            - **Linear Model**: Assumes constant growth (same number of papers added each year)
            - **Exponential Model**: Assumes percentage growth (papers grow by a percentage each year)
            - **R² Score**: Measures how well the model fits the data (1.0 = perfect fit)
            - **Better Model**: The model with higher R² score fits the data better
            """)
        else:
            st.warning("Not enough data points for regression analysis with current selections.")
    
    # ==========================================
    # TAB 4: GEOGRAPHIC ANALYSIS
    # ==========================================
    with tab4:
        st.header("Geographic Distribution")
        st.markdown("Explore research distribution across different countries.")
        
        # Create sub-tabs for different geographic visualizations
        geo_tab1, geo_tab2, geo_tab3 = st.tabs(["Country Rankings", "World Map", "Trends by Country"])
        
        # Filter country data by top N
        df_countries_filtered = df_countries.head(top_n_countries).copy()
        df_countries_filtered.loc[:, 'country_name'] = df_countries_filtered['alpha3_code'].apply(get_country_name)
        
        with geo_tab1:
            st.subheader("Top Research-Producing Countries")
            
            # Debug: Show sample data for treemap
            if st.checkbox("Show sample treemap data (debug)"):
                st.write("Sample treemap data:")
                st.write(f"DataFrame shape: {df_countries_filtered.shape}")
                st.write(f"Columns: {list(df_countries_filtered.columns)}")
                st.write(df_countries_filtered[['country_name', 'paper_count']].head())
                st.write(f"Any null values in country_name: {df_countries_filtered['country_name'].isnull().sum()}")
                st.write(f"Any null values in paper_count: {df_countries_filtered['paper_count'].isnull().sum()}")
                st.write("Data types:")
                st.write(df_countries_filtered[['country_name', 'paper_count']].dtypes)
                st.write(f"Total papers sum: {df_countries_filtered['paper_count'].sum()}")
                st.write(f"Min/Max paper counts: {df_countries_filtered['paper_count'].min()} / {df_countries_filtered['paper_count'].max()}")
            
            # Country treemap
            if not df_countries_filtered.empty and 'country_name' in df_countries_filtered.columns:
                # Clean and prepare treemap data
                treemap_data = df_countries_filtered[
                    df_countries_filtered['country_name'].notna() & 
                    (df_countries_filtered['paper_count'] > 0)
                ].copy()
                
                # Ensure data types are correct
                treemap_data['paper_count'] = treemap_data['paper_count'].astype(int)
                treemap_data['country_name'] = treemap_data['country_name'].astype(str)
                
                # Remove any problematic characters from country names
                treemap_data['country_name'] = treemap_data['country_name'].str.replace(r'[^\w\s-]', '', regex=True)
                
                # Reset index to avoid any indexing issues
                treemap_data = treemap_data.reset_index(drop=True)
                
                if not treemap_data.empty and len(treemap_data) > 0:
                    try:
                        fig_treemap = px.treemap(
                            treemap_data,
                            values='paper_count',
                            names='country_name',
                            parents=[''] * len(treemap_data),  # Add parents
                            title=f"Research Output by Country (Top {top_n_countries})",
                            color='paper_count',
                            color_continuous_scale='Viridis',
                            width=800,
                            height=500
                        )
                        
                        fig_treemap.update_traces(
                            textinfo="label+value",
                            textfont_size=12
                        )
                        
                        fig_treemap.update_layout(
                            template=chart_theme,
                            height=500,
                            margin=dict(t=50, l=25, r=25, b=25)
                        )
                        
                        st.plotly_chart(fig_treemap, use_container_width=True)
                        
                        # Debug info
                        if st.checkbox("Show treemap debug info"):
                            st.write(f"Treemap created successfully with {len(treemap_data)} countries")
                            st.write("Data passed to treemap:")
                            # Use st.write instead of st.dataframe to avoid Arrow issues
                            for i, row in treemap_data[['country_name', 'paper_count']].head(10).iterrows():
                                st.write(f"- {row['country_name']}: {row['paper_count']} papers")
                    
                    except Exception as e:
                        st.error(f"Error creating treemap: {str(e)}")
                        st.write("Attempting alternative bar chart...")
                        
                        # Fallback to bar chart
                        fig_bar = px.bar(
                            treemap_data.head(10),
                            x='paper_count',
                            y='country_name',
                            orientation='h',
                            title=f"Top 10 Countries by Research Output",
                            color='paper_count',
                            color_continuous_scale='Viridis'
                        )
                        fig_bar.update_layout(template=chart_theme, height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.error("No valid data for treemap after filtering")
            else:
                st.error("Unable to create treemap - check data structure")
            
            # Country statistics table
            st.subheader("Country Statistics")
            if not df_countries_filtered.empty:
                country_stats = df_countries_filtered[['country_name', 'paper_count']].copy()
                total_papers = country_stats['paper_count'].sum()
                country_stats['Percentage'] = (country_stats['paper_count'] / total_papers * 100).round(1)
                country_stats.columns = ['Country', 'Papers', 'Percentage (%)']
                st.dataframe(country_stats, use_container_width=True)
            else:
                st.error("No country data available")
        
        with geo_tab2:
            st.subheader("World Research Distribution")
            
            # Prepare data for choropleth map (database now contains alpha-3 codes)
            df_countries_choropleth = df_countries.copy()
            df_countries_choropleth['country_name'] = df_countries_choropleth['alpha3_code'].apply(get_country_name)
            
            # Debug: Show sample data
            if st.checkbox("Show sample map data (debug)"):
                st.write("Sample choropleth data:")
                st.write(df_countries_choropleth[['alpha3_code', 'country_name', 'paper_count']].head(10))
            
            # Create choropleth map using alpha-3 codes
            fig_choropleth = px.choropleth(
                df_countries_choropleth,
                locations='alpha3_code',
                color='paper_count',
                hover_name='country_name',
                hover_data={'alpha3_code': False, 'paper_count': ':,'},
                color_continuous_scale='Viridis',
                title="Global Distribution of Urban Ecology Research",
                labels={'paper_count': 'Number of Papers'},
                range_color=[0, df_countries_choropleth['paper_count'].max()]
            )
            
            fig_choropleth.update_layout(
                template=chart_theme,
                height=600,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    coastlinecolor='rgb(204, 204, 204)',
                )
            )
            
            st.plotly_chart(fig_choropleth, use_container_width=True)
            
            # Map interpretation
            st.markdown("""
            **Map Interpretation:**
            - Darker colors indicate higher research output
            - Hover over countries to see exact paper counts
            - Gray areas represent countries with no data in our dataset
            """)
        
        with geo_tab3:
            st.subheader("Research Trends by Country Over Time")
            
            # Filter country-year data for heatmap
            top_countries = df_countries_filtered['alpha3_code'].tolist()
            df_country_years_filtered = df_country_years[
                (df_country_years['country'].isin(top_countries)) &
                (df_country_years['year'] >= min_year) &
                (df_country_years['year'] <= max_year)
            ]
            
            if not df_country_years_filtered.empty:
                # Create country-year heatmap
                df_country_years_filtered = df_country_years_filtered.copy()
                df_country_years_filtered.loc[:, 'country_name'] = df_country_years_filtered['country'].apply(get_country_name)
                
                # Pivot for heatmap
                heatmap_data = df_country_years_filtered.pivot(
                    index='country_name', 
                    columns='year', 
                    values='paper_count'
                ).fillna(0)
                
                # Create masked version (replace 0 with NaN for better visualization)
                heatmap_data_masked = heatmap_data.replace(0, np.nan)
                
                # Create heatmap
                fig_country_heatmap = px.imshow(
                    heatmap_data_masked,
                    title=f"Research Output Over Time by Country ({min_year}-{max_year})",
                    labels=dict(x="Year", y="Country", color="Papers"),
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                
                fig_country_heatmap.update_layout(
                    template=chart_theme,
                    height=600
                )
                
                st.plotly_chart(fig_country_heatmap, use_container_width=True)
                
                # Time series by country
                st.subheader("Research Growth by Leading Countries")
                
                # Select top 5 countries for time series
                top_5_countries = df_countries_filtered.head(5)
                df_top_countries_time = df_country_years_filtered[
                    df_country_years_filtered['country'].isin(top_5_countries['alpha3_code'])
                ]
                
                if not df_top_countries_time.empty:
                    fig_country_time = px.line(
                        df_top_countries_time,
                        x='year',
                        y='paper_count',
                        color='country_name',
                        title="Research Output Over Time - Top 5 Countries",
                        labels={'year': 'Year', 'paper_count': 'Papers Published', 'country_name': 'Country'}
                    )
                    
                    fig_country_time.update_layout(
                        template=chart_theme,
                        height=500
                    )
                    
                    st.plotly_chart(fig_country_time, use_container_width=True)
        
        # Overall geographic statistics
        st.subheader("Geographic Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Countries Analyzed", len(df_countries_filtered))
        
        with col2:
            st.metric("Leading Country", df_countries_filtered.iloc[0]['country_name'])
        
        with col3:
            total_papers_geo = df_countries_filtered['paper_count'].sum()
            st.metric("Total Papers (Top Countries)", f"{total_papers_geo:,}")
        
        with col4:
            avg_papers_country = df_countries_filtered['paper_count'].mean()
            st.metric("Avg Papers/Country", f"{avg_papers_country:.0f}")
    
    # ==========================================
    # TAB 5: DATA SUMMARY
    # ==========================================
    with tab5:
        st.header("Data Summary")
        st.markdown("Overview of the dataset and current analysis parameters.")
        
        # Analysis parameters
        st.subheader("Current Analysis Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Keywords Selected:**")
            for kw in selected_keywords:
                st.write(f"• {kw}")
        
        with col2:
            st.write("**Analysis Settings:**")
            st.write(f"• Year Range: {min_year} - {max_year}")
            st.write(f"• Top Countries: {top_n_countries}")
            st.write(f"• Chart Theme: {chart_theme}")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        
        # Show sample data
        if st.checkbox("Show sample data"):
            st.write("**Sample keyword data:**")
            st.dataframe(df_filtered.head(10), use_container_width=True)
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Unique Papers", f"{total_unique_papers:,}")
        
        with col2:
            total_countries = len(df_countries)
            st.metric("Countries Represented", total_countries)
        
        with col3:
            year_coverage = 2023 - 1970 + 1
            st.metric("Years Covered", year_coverage)
        
        # Data quality metrics
        st.subheader("Data Quality")
        expanded_instances = len(df_expanded)
        papers_with_countries = df_countries['paper_count'].sum()
        
        st.write(f"• Unique papers in dataset: {total_unique_papers:,}")
        st.write(f"• Keyword instances (papers × keywords): {expanded_instances:,}")
        st.write(f"• Papers with country data: {papers_with_countries:,}")
        st.write(f"• Data completeness: Good coverage across time period")

# ==========================================
# APP ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()

# ==========================================
# REQUIREMENTS.TXT CONTENT (for reference)
# ==========================================
"""
To deploy this app, create a requirements.txt file with:

streamlit
pandas
numpy
plotly
scipy
"""
