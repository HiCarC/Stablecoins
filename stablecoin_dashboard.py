import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Stablecoin Market Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("🏦 Stablecoin Market Analysis for Central Banks")
st.markdown("""
This dashboard provides comprehensive analysis of the stablecoin market, focusing on key metrics 
relevant for central bank policy making and regulatory oversight.
""")

# Load and prepare data
@st.cache_data
def load_data():
    # Generate sample dates
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    # Create sample data with realistic trends
    base_usdt = 70e9  # Starting at 70B
    base_usdc = 40e9  # Starting at 40B
    base_ustc = 10e9  # Starting at 10B
    
    # Add some realistic growth and volatility
    noise = np.random.normal(0, 0.02, len(dates))
    growth = np.linspace(0, 0.5, len(dates))  # 50% growth over period
    
    # Create DataFrame with sample data
    df = pd.DataFrame({
        'Date': dates,
        'USDT': base_usdt * (1 + growth + noise),
        'USDC': base_usdc * (1 + growth * 0.8 + noise),
        'USTC': base_ustc * (1 + growth * 0.3 + noise)
    })
    
    # Calculate total market cap
    df['Total'] = df[['USDT', 'USDC', 'USTC']].sum(axis=1)
    
    # Calculate market shares
    df['USDT_Share'] = (df['USDT'] / df['Total']) * 100
    df['USDC_Share'] = (df['USDC'] / df['Total']) * 100
    df['USTC_Share'] = (df['USTC'] / df['Total']) * 100
    
    # Calculate Herfindahl index
    df['herfindahl_index'] = (
        (df['USDT_Share']/100)**2 + 
        (df['USDC_Share']/100)**2 + 
        (df['USTC_Share']/100)**2
    )
    
    return df

# Load the data
df = load_data()

# Add this function after load_data()
@st.cache_data
def load_fed_data():
    # Load FED data
    fed_data = pd.read_csv('FRB_H15.csv', skiprows=5)
    fed_data['Time Period'] = pd.to_datetime(fed_data['Time Period'])
    
    # Select relevant columns
    cols = ['Time Period', 
            'RIFSPFF_N.M',  # Federal funds rate
            'RIFLGFCY10_N.M',  # 10-year Treasury
            'RIFLGFCY02_N.M']  # 2-year Treasury
    
    fed_df = fed_data[cols].copy()
    fed_df.columns = ['Date', 'Fed_Funds_Rate', 'Treasury_10Y', 'Treasury_2Y']
    return fed_df

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Market Overview", "Concentration Analysis", "Stability Metrics", 
     "Monetary Policy Impact", "Detailed Analytics", "Policy Recommendations"]
)

if section == "Market Overview":
    st.header("📈 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Market Size Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Total']/1e9)
        ax.set_title("Total Stablecoin Market Cap (Billions USD)")
        ax.grid(True)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Market Share Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_latest = df.iloc[-1]
        shares = [df_latest['USDT_Share'], df_latest['USDC_Share'], df_latest['USTC_Share']]
        labels = ['USDT', 'USDC', 'USTC']
        ax.pie(shares, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

elif section == "Concentration Analysis":
    st.header("🎯 Market Concentration Analysis")
    
    # HHI Evolution
    st.subheader("Herfindahl Index Evolution")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['herfindahl_index'])
    ax.axhline(y=0.25, color='r', linestyle='--', label='High Concentration Threshold')
    ax.set_title("Market Concentration Over Time")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    # Risk Level Analysis
    monthly_hhi = df.groupby(df['Date'].dt.to_period('M'))['herfindahl_index'].mean()
    risk_levels = {
        'Low': (monthly_hhi < 0.15).mean() * 100,
        'Moderate': ((monthly_hhi >= 0.15) & (monthly_hhi < 0.25)).mean() * 100,
        'High': (monthly_hhi >= 0.25).mean() * 100
    }
    
    st.subheader("Risk Level Distribution")
    st.bar_chart(risk_levels)

elif section == "Stability Metrics":
    st.header("📊 Stability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supply Volatility
        volatility = df['Total'].pct_change().rolling(30).std() * 100
        st.subheader("30-Day Rolling Supply Volatility")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], volatility)
        ax.set_title("Supply Volatility (%)")
        ax.grid(True)
        st.pyplot(fig)
    
    with col2:
        # Growth Rate
        growth_rate = df['Total'].pct_change().rolling(30).mean() * 100
        st.subheader("30-Day Rolling Growth Rate")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], growth_rate)
        ax.set_title("Growth Rate (%)")
        ax.grid(True)
        st.pyplot(fig)

elif section == "Monetary Policy Impact":
    st.header("🏦 Monetary Policy Impact Analysis")
    
    # Load FED data
    fed_df = load_fed_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot Fed Funds Rate vs Stablecoin Market Cap
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot total market cap
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Stablecoin Market Cap (Billions USD)', color='tab:blue')
        ax1.plot(df['Date'], df['Total']/1e9, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis for Fed Funds Rate
        ax2 = ax1.twinx()
        ax2.set_ylabel('Fed Funds Rate (%)', color='tab:red')
        ax2.plot(fed_df['Date'], fed_df['Fed_Funds_Rate'], color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Stablecoin Market Cap vs Fed Funds Rate')
        st.pyplot(fig)
        
    with col2:
        # Plot Yield Curve Evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        spread = fed_df['Treasury_10Y'] - fed_df['Treasury_2Y']
        ax.plot(fed_df['Date'], spread, label='10Y-2Y Spread')
        ax.axhline(y=0, color='r', linestyle='--', label='Inversion Line')
        ax.set_title('Treasury Yield Curve Spread (10Y-2Y)')
        ax.set_ylabel('Spread (%)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Additional Analysis
    col3, col4 = st.columns(2)
    
    with col3:
        # Calculate correlation between Fed Funds Rate and Market Cap Growth
        merged_df = pd.merge(
            df, 
            fed_df, 
            on='Date', 
            how='inner'
        )
        
        growth_rate = merged_df['Total'].pct_change().rolling(30).mean() * 100
        correlation = growth_rate.corr(merged_df['Fed_Funds_Rate'])
        
        st.metric(
            "Correlation: Growth Rate vs Fed Funds Rate",
            f"{correlation:.2f}",
            delta=None,
            delta_color="normal"
        )
        
    with col4:
        # Calculate average market cap during different rate regimes
        low_rate = merged_df[merged_df['Fed_Funds_Rate'] < 2]['Total'].mean() / 1e9
        high_rate = merged_df[merged_df['Fed_Funds_Rate'] >= 2]['Total'].mean() / 1e9
        
        st.metric(
            "Avg Market Cap: High Rates vs Low Rates",
            f"${high_rate:.1f}B vs ${low_rate:.1f}B",
            delta=f"{((high_rate/low_rate)-1)*100:.1f}%",
            delta_color="normal"
        )
    
    # Risk Analysis
    st.subheader("Monetary Policy Risk Analysis")
    
    # Calculate current metrics
    latest_spread = spread.iloc[-1]
    latest_rate = fed_df['Fed_Funds_Rate'].iloc[-1]
    
    col5, col6 = st.columns(2)
    
    with col5:
        if latest_spread < 0:
            st.warning("⚠️ Yield Curve Inversion Detected")
            st.markdown("""
            **Potential Impacts:**
            - Increased redemption risk
            - Possible flight to quality
            - Need for enhanced liquidity monitoring
            """)
    
    with col6:
        if latest_rate > 4:
            st.info("ℹ️ High Rate Environment")
            st.markdown("""
            **Considerations:**
            - Yield competition from traditional assets
            - Potential pressure on stablecoin reserves
            - Need for competitive yield strategies
            """)

elif section == "Detailed Analytics":
    st.header("📊 Detailed Analytics")
    
    # Time Series Decomposition
    st.subheader("Market Cap Time Series Decomposition")
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        for coin in ['USDT', 'USDC', 'USTC']:
            ax.plot(df['Date'], df[coin]/1e9, label=coin)
        ax.set_title("Individual Stablecoin Market Caps")
        ax.set_ylabel("Billions USD")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        # Market Share Evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stackplot(df['Date'], 
                    df['USDT_Share'],
                    df['USDC_Share'],
                    df['USTC_Share'],
                    labels=['USDT', 'USDC', 'USTC'])
        ax.set_title("Market Share Evolution")
        ax.set_ylabel("Percentage")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Calculate correlations
        corr_matrix = df[['USDT', 'USDC', 'USTC']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Stablecoin Correlation Matrix")
        st.pyplot(fig)
    
    with col4:
        # Rolling Correlation
        window = 30
        rolling_corr = df['USDT'].rolling(window).corr(df['USDC'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'][window-1:], rolling_corr)
        ax.set_title(f"{window}-Day Rolling Correlation (USDT-USDC)")
        ax.grid(True)
        st.pyplot(fig)
    
    # Volume and Volatility Analysis
    st.subheader("Volume and Volatility Patterns")
    col5, col6 = st.columns(2)
    
    with col5:
        # Daily Changes Distribution
        daily_changes = df['Total'].pct_change().dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(daily_changes, kde=True, ax=ax)
        ax.set_title("Distribution of Daily Market Cap Changes")
        ax.set_xlabel("Daily Change (%)")
        st.pyplot(fig)
    
    with col6:
        # Volatility Clustering
        fig, ax = plt.subplots(figsize=(10, 6))
        abs_returns = abs(daily_changes)
        ax.scatter(abs_returns.index, abs_returns, alpha=0.5)
        ax.set_title("Volatility Clustering")
        ax.set_ylabel("Absolute Daily Change (%)")
        st.pyplot(fig)
    
    # Market Concentration Metrics
    st.subheader("Market Concentration Metrics")
    col7, col8 = st.columns(2)
    
    with col7:
        # Top Players Market Share
        fig, ax = plt.subplots(figsize=(10, 6))
        top_2_share = df['USDT_Share'] + df['USDC_Share']
        ax.plot(df['Date'], top_2_share)
        ax.set_title("Combined Market Share of Top 2 Stablecoins")
        ax.set_ylabel("Market Share (%)")
        ax.grid(True)
        st.pyplot(fig)
    
    with col8:
        # Market Concentration Trend
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_hhi = df['herfindahl_index'].rolling(30).mean()
        ax.plot(df['Date'], rolling_hhi)
        ax.set_title("30-Day Rolling Average HHI")
        ax.grid(True)
        st.pyplot(fig)

else:  # Policy Recommendations
    st.header("📋 Policy Recommendations")
    
    # Calculate latest metrics for recommendations
    latest_hhi = df['herfindahl_index'].iloc[-1]
    market_size = df['Total'].iloc[-1] / 1e9
    volatility_trend = "Deteriorating" if df['Total'].pct_change().rolling(30).std().iloc[-30:].mean() > \
                      df['Total'].pct_change().rolling(30).std().mean() else "Improving"
    
    # Display recommendations
    if latest_hhi > 0.25:
        st.error("🚨 HIGH PRIORITY: Market concentration exceeds critical threshold")
        st.markdown("""
        **Recommended Actions:**
        - Implement market concentration limits
        - Enhance monitoring of dominant players
        - Consider measures to promote competition
        """)
    
    if market_size > 100:
        st.warning(f"⚠️ CRITICAL: Market size (${market_size:.1f}B) requires comprehensive oversight")
        st.markdown("""
        **Recommended Actions:**
        - Develop specific oversight framework
        - Implement systemic risk monitoring
        - Consider capital requirements for major issuers
        """)
    
    if volatility_trend == "Deteriorating":
        st.info("ℹ️ ATTENTION: Supply stability showing concerning trends")
        st.markdown("""
        **Recommended Actions:**
        - Implement supply growth limits
        - Enhance reporting requirements
        - Develop early warning systems
        """)

# Footer
st.markdown("---")
st.markdown("""
*This dashboard is designed to support central bank decision-making regarding stablecoin regulation and oversight.*
""")