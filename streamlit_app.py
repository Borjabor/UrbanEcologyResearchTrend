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
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import pycountry

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

@st.cache_data  # Streamlit decorator to cache data for better performance
def load_data():
    """
    Load data from the SQLite database and parse the keyword combinations.
    The @st.cache_data decorator ensures this only runs once per session.
    """
    try:
        conn = sqlite3.connect('papers.db')
        
        # Load all papers first
        df_all_papers = pd.read_sql_query("""
            SELECT paperId, year, search_keyword, title, authors, firstAuthorCountryIso, citationCount
            FROM papers 
            WHERE search_keyword IS NOT NULL AND search_keyword != ''
        """, conn)
        
        # Define the 6 original keywords
        original_keywords = [
            'urban ecology',
            'urban biodiversity', 
            'urban ecosystem',
            'urban green spaces',
            'urban vegetation',
            'urban wildlife'
        ]
        
        # Parse keyword combinations and create individual records
        keyword_data = []
        for _, row in df_all_papers.iterrows():
            keywords_in_paper = [kw.strip() for kw in row['search_keyword'].split(',')]
            for keyword in keywords_in_paper:
                if keyword in original_keywords:
                    keyword_data.append({
                        'paperId': row['paperId'],
                        'year': row['year'],
                        'search_keyword': keyword,
                        'title': row['title'],
                        'authors': row['authors'],
                        'firstAuthorCountryIso': row['firstAuthorCountryIso'],
                        'citationCount': row['citationCount']
                    })
        
        # Create expanded dataset with individual keyword records
        df_expanded = pd.DataFrame(keyword_data)
        
        # Create keyword counts by year (count papers per keyword per year)
        df_keywords = df_expanded.groupby(['search_keyword', 'year']).size().reset_index()
        df_keywords.columns = ['search_keyword', 'year', 'paper_count']
        
        # Load country data (now using alpha-3 codes directly from database)
        df_countries = pd.read_sql_query("""
            SELECT firstAuthorCountryIso as alpha3_code, COUNT(*) as paper_count
            FROM papers 
            WHERE firstAuthorCountryIso IS NOT NULL AND firstAuthorCountryIso != ''
            GROUP BY firstAuthorCountryIso 
            ORDER BY paper_count DESC
        """, conn)
        
        # Load country-year data
        df_country_years = pd.read_sql_query("""
            SELECT 
                year,
                firstAuthorCountryIso as country,
                COUNT(*) as paper_count
            FROM papers 
            WHERE firstAuthorCountryIso IS NOT NULL AND firstAuthorCountryIso != '' 
            GROUP BY year, firstAuthorCountryIso
            ORDER BY year, firstAuthorCountryIso
        """, conn)
        
        conn.close()
        
        return df_keywords, df_countries, df_expanded, df_country_years
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

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
        if keyword == "📊 Total (All Keywords)":
            actual_keywords.append("Total (All Keywords)")
        else:
            actual_keywords.append(keyword)
    return actual_keywords


def perform_linear_regression(df_filtered, selected_keywords):
    """
    Perform linear regression analysis on the selected keywords and year range.
    Returns results dataframe with statistics for each keyword.
    """
    results = []
    
    # Convert display keywords to actual DataFrame keywords
    actual_keywords = get_actual_keywords(selected_keywords)
    
    for keyword in actual_keywords:
        # Filter data for this keyword
        keyword_data = df_filtered[df_filtered['search_keyword'] == keyword].sort_values('year')
        
        if len(keyword_data) < 3:  # Need at least 3 points for regression
            continue
            
        years = keyword_data['year'].values
        counts = keyword_data['paper_count'].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
        
        results.append({
            'keyword': keyword,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)


def perform_log_regression(df_filtered, selected_keywords):
    """
    Perform logarithmic regression to test for exponential growth patterns.
    """
    results = []
    
    # Convert display keywords to actual DataFrame keywords
    actual_keywords = get_actual_keywords(selected_keywords)
    
    for keyword in actual_keywords:
        # Filter data for this keyword
        keyword_data = df_filtered[df_filtered['search_keyword'] == keyword].sort_values('year')
        
        if len(keyword_data) < 3:  # Need at least 3 points for regression
            continue
            
        years = keyword_data['year'].values
        counts = keyword_data['paper_count'].values
        
        # Add small constant to avoid log(0)
        log_counts = np.log(counts + 1)
        
        # Perform linear regression on log-transformed data
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_counts)
        
        # Calculate annual growth rate from log regression
        annual_growth_rate = (np.exp(slope) - 1) * 100
        doubling_time = np.log(2) / slope if slope > 0 else np.inf
        
        results.append({
            'keyword': keyword,
            'slope_log': slope,
            'intercept_log': intercept,
            'r_squared_log': r_value**2,
            'p_value_log': p_value,
            'annual_growth_rate_percent': annual_growth_rate,
            'doubling_time_years': doubling_time,
            'significant_log': p_value < 0.05
        })
    
    return pd.DataFrame(results)


def compare_linear_vs_exponential(df_filtered, selected_keywords):
    """
    Compare linear vs exponential growth models for keywords.
    """
    linear_results = perform_linear_regression(df_filtered, selected_keywords)
    log_results = perform_log_regression(df_filtered, selected_keywords)
    
    if linear_results.empty or log_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Merge results
    comparison = pd.merge(
        linear_results[['keyword', 'r_squared', 'p_value', 'significant']], 
        log_results[['keyword', 'r_squared_log', 'p_value_log', 'significant_log', 
                    'annual_growth_rate_percent', 'doubling_time_years']], 
        on='keyword', 
        suffixes=('_linear', '_log')
    )
    
    comparison['better_fit'] = np.where(
        comparison['r_squared_log'] > comparison['r_squared'], 
        'Exponential', 'Linear'
    )
    comparison['r_squared_difference'] = comparison['r_squared_log'] - comparison['r_squared']
    
    return comparison, linear_results, log_results

def create_keyword_similarity_matrix(df_expanded, selected_keywords, year_range):
    """
    Create keyword co-occurrence similarity matrix based on papers that contain
    multiple keywords. Shows which keywords appear together in the same papers.
    Note: Excludes 'Total' option as it represents aggregated data.
    """
    # Filter out the Total option for similarity analysis
    analysis_keywords = [kw for kw in selected_keywords if kw != "📊 Total (All Keywords)"]
    
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
    st.title("🏙️ Urban Ecology Research Trends")
    st.markdown("""
    **Interactive Dashboard for Urban Ecology Research Analysis**
    
    This dashboard allows you to explore trends in urban ecology research publications 
    from 1970-2023. Customize your analysis by selecting keywords, adjusting year ranges, 
    and modifying various parameters.
    """)
    
    # Load data
    with st.spinner("Loading data..."):  # Show loading spinner
        df_keywords, df_countries, df_expanded, df_country_years = load_data()
    
    if df_keywords is None:
        st.error("Failed to load data. Please check that 'papers.db' exists in the same directory.")
        return
    
    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Get available keywords
    available_keywords = sorted(df_keywords['search_keyword'].unique())
    keyword_options = ["📊 Total (All Keywords)"] + available_keywords
    
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
        st.warning("⚠️ Please select at least one keyword to begin analysis.")
        return
    
    # Filter data based on selections
    # Handle "Total" option separately
    if "📊 Total (All Keywords)" in selected_keywords:
        # If Total is selected, include all keywords for aggregation
        actual_keywords = available_keywords
        df_filtered = df_keywords[
            (df_keywords['search_keyword'].isin(actual_keywords)) &
            (df_keywords['year'] >= min_year) &
            (df_keywords['year'] <= max_year)
        ]
        
        # Create aggregated data for total
        df_total = df_filtered.groupby('year')['paper_count'].sum().reset_index()
        df_total['search_keyword'] = 'Total (All Keywords)'
        
        # If only Total is selected, use just the total data
        if len(selected_keywords) == 1:
            df_filtered = df_total
        else:
            # Include both individual keywords and total
            individual_keywords = [kw for kw in selected_keywords if kw != "📊 Total (All Keywords)"]
            df_individual = df_keywords[
                (df_keywords['search_keyword'].isin(individual_keywords)) &
                (df_keywords['year'] >= min_year) &
                (df_keywords['year'] <= max_year)
            ]
            df_filtered = pd.concat([df_individual, df_total], ignore_index=True)
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
        "📊 Time Series", 
        "🔗 Keyword Relationships", 
        "📈 Regression Analysis",
        "🌍 Geographic Analysis",
        "📋 Data Summary"
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
            total_papers = df_filtered['paper_count'].sum()
            st.metric("Total Papers", f"{total_papers:,}")
        
        with col2:
            avg_per_year = df_filtered.groupby('year')['paper_count'].sum().mean()
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
        analysis_keywords = [kw for kw in selected_keywords if kw != "📊 Total (All Keywords)"]
        
        if len(analysis_keywords) < 2:
            if "📊 Total (All Keywords)" in selected_keywords:
                st.info("💡 Keyword relationship analysis is not applicable when 'Total (All Keywords)' is selected, as it represents aggregated data. Please select at least 2 individual keywords to see their relationships.")
            else:
                st.warning("⚠️ Please select at least 2 keywords to analyze relationships.")
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
                    color_continuous_scale="Viridis"
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
            comparison, linear_results, log_results = compare_linear_vs_exponential(df_filtered, selected_keywords)
        
        if not comparison.empty:
            # Display comparison results
            st.subheader("Model Comparison: Linear vs Exponential")
            
            # Format the results for display
            display_comparison = comparison.copy()
            display_comparison['Annual Growth Rate (%)'] = display_comparison['annual_growth_rate_percent'].round(2)
            display_comparison['Doubling Time (years)'] = display_comparison['doubling_time_years'].round(1)
            display_comparison['Linear R²'] = display_comparison['r_squared'].round(3)
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
                    y=comparison['r_squared'],
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
                    colors = ['green' if sig else 'red' for sig in finite_growth['significant_log']]
                    
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
                    log_pred = np.exp(log_kw['slope_log'] * year_range_extended + log_kw['intercept_log']) - 1
                    
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
                        name=f'Exponential (R²={log_kw["r_squared_log"]:.3f})',
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
                        st.write(f"• Significant: {'Yes' if linear_kw['significant'] else 'No'}")
                    
                    with col2:
                        st.markdown("**Exponential Model:**")
                        st.write(f"• Growth rate: {log_kw['annual_growth_rate_percent']:.1f}%/year")
                        st.write(f"• Doubling time: {log_kw['doubling_time_years']:.1f} years")
                        st.write(f"• R² score: {log_kw['r_squared_log']:.3f}")
                        st.write(f"• P-value: {log_kw['p_value_log']:.2e}")
                        st.write(f"• Significant: {'Yes' if log_kw['significant_log'] else 'No'}")
            
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
        geo_tab1, geo_tab2, geo_tab3 = st.tabs(["📊 Country Rankings", "🗺️ World Map", "📈 Trends by Country"])
        
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
                
                # Create heatmap
                fig_country_heatmap = px.imshow(
                    heatmap_data,
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
            total_unique_papers = len(df_expanded)
            st.metric("Total Unique Papers", f"{total_unique_papers:,}")
        
        with col2:
            total_countries = len(df_countries)
            st.metric("Countries Represented", total_countries)
        
        with col3:
            year_coverage = 2023 - 1970 + 1
            st.metric("Years Covered", year_coverage)
        
        # Data quality metrics
        st.subheader("Data Quality")
        papers_with_keywords = len(df_expanded)
        papers_with_countries = df_countries['paper_count'].sum()
        
        st.write(f"• Papers with keyword data: {papers_with_keywords:,}")
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
