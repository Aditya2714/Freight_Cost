import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load trained models and preprocessor."""
    regressor = joblib.load('models/freight_regressor.pkl')
    classifier = joblib.load('models/risk_classifier.pkl')
    preprocessor_data = joblib.load('models/preprocessor.pkl')
    return regressor, classifier, preprocessor_data

@st.cache_data
def load_data():
    """Load featured invoice data."""
    return pd.read_csv('data/featured_invoices.csv')

def main():
    st.title("📊 Vendor Invoice Intelligence System")
    st.markdown("End-to-end ML system for freight cost prediction and invoice risk detection")
    
    # Load models and data
    try:
        regressor, classifier, preprocessor_data = load_models()
        df = load_data()
    except FileNotFoundError:
        st.error("Models not found. Please run the training pipeline first.")
        st.code("python src/generate_data.py\npython src/database_manager.py\npython src/data_preprocessing.py\npython src/train_model.py")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📈 Predictions", "🔍 Model Insights", "📋 Data Explorer", "📊 Presentation Dashboard"])
    
    if page == "📊 Presentation Dashboard":
        show_presentation_dashboard(df)
    elif page == "🏠 Dashboard":
        show_dashboard(df)
    elif page == "📈 Predictions":
        show_predictions(regressor, classifier, preprocessor_data, df)
    elif page == "🔍 Model Insights":
        show_model_insights(regressor, classifier, df, preprocessor_data)
    elif page == "📋 Data Explorer":
        show_data_explorer(df)

def show_presentation_dashboard(df):
    """Presentation-ready dashboard with all key visualizations."""
    
    st.title("📊 Presentation Dashboard")
    st.markdown("### Executive Summary for Leadership")
    
    # Business Impact Metrics
    st.subheader("Business Impact")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Money saved calculation
        risky_invoices = df['is_risky'].sum()
        avg_fraud_amount = df[df['is_risky'] == 1]['freight_cost'].mean()
        manual_catch_rate = 0.60
        ml_catch_rate = 0.95
        
        money_at_risk = risky_invoices * avg_fraud_amount * manual_catch_rate
        money_protected = risky_invoices * avg_fraud_amount * ml_catch_rate
        money_saved = money_protected - money_at_risk
        
        st.metric("💰 Money Saved", f"${money_saved:,.0f}", "From fraud prevention")
    
    with col2:
        # Time saved calculation
        total_invoices = len(df)
        minutes_per_invoice_manual = 5
        risky_count = risky_invoices
        minutes_per_risky_manual = 5
        
        manual_hours = (total_invoices * minutes_per_invoice_manual) / 60
        ml_hours = (risky_count * minutes_per_risky_manual) / 60
        hours_saved = manual_hours - ml_hours
        
        st.metric("⏱️ Time Saved", f"{hours_saved:,.0f} Hours/Month", "Manual review eliminated")
    
    with col3:
        # Efficiency gain
        efficiency_gain = ((manual_hours - ml_hours) / manual_hours) * 100
        st.metric("📈 Efficiency Gain", f"{efficiency_gain:.1f}%", "Automation impact")
    
    st.caption("**Business Impact:** Our ML system prevents fraud and eliminates manual review, saving money and time every month.")
    
    st.divider()
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Monthly Freight Cost Trends")
        
        # Create monthly aggregation
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        df['month'] = df['invoice_date'].dt.to_period('M')
        monthly_costs = df.groupby('month')['freight_cost'].sum().reset_index()
        monthly_costs['month'] = monthly_costs['month'].astype(str)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_costs['month'], monthly_costs['freight_cost'], 
                marker='o', linewidth=2, markersize=8, color='#3498db')
        ax.fill_between(range(len(monthly_costs)), monthly_costs['freight_cost'], 
                       alpha=0.3, color='#3498db')
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Total Freight Cost ($)", fontsize=12)
        ax.set_title("Freight Spending Over Time", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** Freight costs month by month. Look for spikes to identify when costs increased. Helps budget for future quarters.")
    
    with col2:
        st.subheader("🥧 Cost Distribution by Vendor")
        
        vendor_costs = df.groupby('vendor_name')['freight_cost'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
        wedges, texts, autotexts = ax.pie(vendor_costs.values, labels=vendor_costs.index, 
                                           autopct='%1.1f%%', colors=colors[:len(vendor_costs)],
                                           startangle=90, pctdistance=0.85)
        ax.set_title("Spending by Vendor", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** Which vendors receive the most of our shipping budget. Large slices may need contract renegotiation for better rates.")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Cost Breakdown Waterfall")
        
        # Calculate cost components
        base_cost = (df['distance_km'] * df['base_rate_per_km'] * (df['weight_kg'] / 100)).mean()
        fuel_cost = base_cost * (df['fuel_surcharge_pct'].mean() / 100)
        volume_cost = df['volume_m3'].mean() * 100
        handling = df['handling_fee'].mean()
        expedited_cost = df['is_expedited'].mean() * base_cost * 0.3
        customs_cost = df['customs_clearance'].mean() * 250
        total = base_cost + fuel_cost + volume_cost + handling + expedited_cost + customs_cost
        
        categories = ['Base Cost', 'Fuel', 'Volume', 'Handling', 'Expedited', 'Customs', 'Total']
        values = [base_cost, fuel_cost, volume_cost, handling, expedited_cost, customs_cost, total]
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c', '#2c3e50']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel("Cost ($)", fontsize=12)
        ax.set_title("Average Cost Breakdown Per Invoice", fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                   f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** How each invoice cost adds up. Base shipping is largest - fuel surcharge and expedited fees are areas to negotiate.")
    
    with col2:
        st.subheader("⚠️ Risk Analysis by Vendor")
        
        vendor_risk = df.groupby('vendor_name').agg({
            'is_risky': ['mean', 'sum'],
            'freight_cost': 'sum'
        }).round(3)
        vendor_risk.columns = ['Risk Rate', 'Risky Count', 'Total Spend']
        vendor_risk = vendor_risk.sort_values('Risk Rate', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(vendor_risk.index, vendor_risk['Risk Rate'] * 100, 
                       color=['#e74c3c' if x > 0.5 else '#f39c12' if x > 0.3 else '#2ecc71' 
                              for x in vendor_risk['Risk Rate']])
        ax.set_xlabel("Risk Rate (%)", fontsize=12)
        ax.set_title("Invoice Risk Rate by Vendor", fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, vendor_risk['Risk Rate'] * 100):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** Percentage of risky invoices per vendor. Red bars (>50%) need immediate audit attention to prevent fraud.")
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Risk Trends Over Time")
        
        monthly_risk = df.groupby('month')['is_risky'].mean().reset_index()
        monthly_risk['month'] = monthly_risk['month'].astype(str)
        monthly_risk['is_risky'] = monthly_risk['is_risky'] * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_risk['month'], monthly_risk['is_risky'], 
                marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.fill_between(range(len(monthly_risk)), monthly_risk['is_risky'], 
                       alpha=0.3, color='#e74c3c')
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Risk Rate (%)", fontsize=12)
        ax.set_title("Fraud Risk Trend Over Time", fontsize=14, fontweight='bold')
        ax.axhline(y=df['is_risky'].mean() * 100, color='gray', linestyle='--', 
                   label=f'Average: {df["is_risky"].mean()*100:.1f}%')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** Is fraud increasing or decreasing over time? Rising trend means we need stronger controls. Falling trend means our system is working.")
    
    with col2:
        st.subheader("🗺️ Cost by Route")
        
        route_stats = df.groupby(['origin_city', 'destination_city'])['freight_cost'].mean().reset_index()
        route_stats = route_stats.sort_values('freight_cost', ascending=False).head(10)
        route_stats['route'] = route_stats['origin_city'] + ' → ' + route_stats['destination_city']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(route_stats['route'], route_stats['freight_cost'], color='#3498db')
        ax.set_xlabel("Average Cost ($)", fontsize=12)
        ax.set_title("Top 10 Most Expensive Routes", fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, route_stats['freight_cost']):
            ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2, 
                   f'${val:,.0f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("**What this shows:** Most expensive shipping routes. Consider alternative carriers or routes for these paths to reduce costs.")
    
    st.divider()
    
    # Key Insights Summary
    st.subheader("📋 Key Insights for Leadership")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Cost Insights:**
        - Total freight spend: ${:,.0f}
        - Average cost per invoice: ${:,.0f}
        - Most expensive carrier: {} (${:,.0f} avg)
        - Top route: {} → {}
        """.format(
            df['freight_cost'].sum(),
            df['freight_cost'].mean(),
            df.groupby('carrier_type')['freight_cost'].mean().idxmax(),
            df.groupby('carrier_type')['freight_cost'].mean().max(),
            df.groupby(['origin_city', 'destination_city'])['freight_cost'].mean().idxmax()[0],
            df.groupby(['origin_city', 'destination_city'])['freight_cost'].mean().idxmax()[1]
        ))
    
    with col2:
        st.warning("""
        **Risk Insights:**
        - Risky invoices: {} ({:.1f}%)
        - Highest risk vendor: {} ({:.1f}%)
        - Estimated fraud prevention: ${:,.0f}
        - Recommendation: Audit top 3 risky vendors
        """.format(
            df['is_risky'].sum(),
            df['is_risky'].mean() * 100,
            df.groupby('vendor_name')['is_risky'].mean().idxmax(),
            df.groupby('vendor_name')['is_risky'].mean().max() * 100,
            df['is_risky'].sum() * df[df['is_risky']==1]['freight_cost'].mean() * 0.35
        ))

def show_dashboard(df):
    """Display main dashboard with KPIs."""
    
    st.header("Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invoices", f"{len(df):,}")
    
    with col2:
        st.metric("Avg Freight Cost", f"${df['freight_cost'].mean():,.2f}")
    
    with col3:
        st.metric("Risky Invoices", f"{df['is_risky'].sum():,}", 
                  f"{df['is_risky'].mean()*100:.1f}%")
    
    with col4:
        st.metric("Total Freight Spend", f"${df['freight_cost'].sum():,.0f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Freight Cost Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['freight_cost'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Freight Cost ($)")
        ax.set_ylabel("Count")
        ax.axvline(df['freight_cost'].mean(), color='red', linestyle='--', label=f"Mean: ${df['freight_cost'].mean():,.0f}")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Risk Distribution by Carrier")
        risk_by_carrier = df.groupby('carrier_type')['is_risky'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_by_carrier.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c', '#3498db'])
        ax.set_ylabel("Risk Rate")
        ax.set_xlabel("Carrier Type")
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)
    
    # Vendor analysis
    st.subheader("Top Vendors by Freight Spend")
    vendor_stats = df.groupby('vendor_name').agg({
        'freight_cost': ['sum', 'mean', 'count'],
        'is_risky': 'mean'
    }).round(2)
    vendor_stats.columns = ['Total Spend', 'Avg Cost', 'Transactions', 'Risk Rate']
    st.dataframe(vendor_stats.sort_values('Total Spend', ascending=False))

def show_predictions(regressor, classifier, preprocessor_data, df):
    """Interactive prediction interface."""
    
    st.header("Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Invoice Details")
        
        vendor = st.selectbox("Vendor", sorted(df['vendor_name'].unique()))
        
        product = st.selectbox("Product Category", sorted(df['product_category'].unique()))
        
        origin = st.selectbox("Origin City", sorted(df['origin_city'].unique()))
        destination = st.selectbox("Destination City", sorted(df['destination_city'].unique()))
        
        carrier = st.selectbox("Carrier Type", sorted(df['carrier_type'].unique()))
    
    with col2:
        st.subheader("Shipment Details")
        
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5000.0, value=100.0)
        volume = st.number_input("Volume (m³)", min_value=0.01, max_value=50.0, value=1.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=50)
        distance = st.number_input("Distance (km)", min_value=100, max_value=15000, value=2000)
        
        base_rate = st.slider("Base Rate ($/km)", 0.5, 5.0, 2.0)
        fuel_surcharge = st.slider("Fuel Surcharge (%)", 5.0, 25.0, 12.0)
        insurance = st.number_input("Insurance Value ($)", min_value=100, max_value=500000, value=10000)
        
        expedited = st.checkbox("Expedited Delivery")
        customs = st.checkbox("Customs Clearance Required")
    
    if st.button("Predict Freight Cost & Risk", type="primary"):
        # Look up aggregated features from the dataset
        vendor_stats = df[df['vendor_name'] == vendor].agg({
            'freight_cost': ['mean', 'std'],
            'delivery_days': 'mean',
            'invoice_id': 'count'
        })
        avg_vendor_cost = vendor_stats['freight_cost']['mean']
        std_vendor_cost = vendor_stats['freight_cost']['std'] if vendor_stats['freight_cost']['std'] > 0 else avg_vendor_cost * 0.1
        vendor_avg_delivery = vendor_stats['delivery_days']['mean']
        vendor_transaction_count = vendor_stats['invoice_id']['count']
        
        route_data = df[(df['origin_city'] == origin) & (df['destination_city'] == destination)]
        if len(route_data) > 0:
            avg_route_cost = route_data['freight_cost'].mean()
            avg_route_distance = route_data['distance_km'].mean()
            route_count = len(route_data)
        else:
            avg_route_cost = df['freight_cost'].mean()
            avg_route_distance = df['distance_km'].mean()
            route_count = 0
        
        product_data = df[df['product_category'] == product]
        avg_product_weight = product_data['weight_kg'].mean()
        avg_product_volume = product_data['volume_m3'].mean()
        avg_product_freight = product_data['freight_cost'].mean()
        
        # Create input dataframe with ALL required features
        base_freight = distance * base_rate * (weight / 100)
        
        input_data = pd.DataFrame({
            'vendor_name': [vendor],
            'product_category': [product],
            'origin_city': [origin],
            'destination_city': [destination],
            'carrier_type': [carrier],
            'weight_kg': [weight],
            'volume_m3': [volume],
            'quantity': [quantity],
            'distance_km': [distance],
            'base_rate_per_km': [base_rate],
            'fuel_surcharge_pct': [fuel_surcharge],
            'insurance_value': [insurance],
            'handling_fee': [50],
            'delivery_days': [5],
            'is_expedited': [int(expedited)],
            'customs_clearance': [int(customs)],
            # Computed cost features
            'cost_per_unit': [base_freight / max(quantity, 1)],
            'cost_per_km': [base_freight / max(distance, 1)],
            'cost_per_kg': [base_freight / max(weight, 0.1)],
            'density': [weight / max(volume, 0.01)],
            # Date features
            'invoice_month': [datetime.now().month],
            'invoice_day': [datetime.now().day],
            'invoice_day_of_week': [datetime.now().weekday()],
            # Aggregated features from dataset
            'avg_vendor_cost': [avg_vendor_cost],
            'std_vendor_cost': [std_vendor_cost],
            'vendor_transaction_count': [vendor_transaction_count],
            'vendor_avg_delivery': [vendor_avg_delivery],
            'avg_route_cost': [avg_route_cost],
            'avg_route_distance': [avg_route_distance],
            'route_count': [route_count],
            'avg_product_weight': [avg_product_weight],
            'avg_product_volume': [avg_product_volume],
            'avg_product_freight': [avg_product_freight],
            # Deviation features
            'vendor_cost_zscore': [(base_freight - avg_vendor_cost) / max(std_vendor_cost, 1)],
            'route_cost_deviation': [(base_freight - avg_route_cost) / max(avg_route_cost, 1)],
            # Risk indicators
            'high_value_flag': [1 if insurance > 100000 else 0],
            'low_quantity_flag': [1 if quantity < 10 else 0],
            'above_avg_cost_flag': [1 if base_freight > avg_vendor_cost * 1.5 else 0],
        })
        
        # Get predictions
        try:
            preprocessor = preprocessor_data['preprocessor']
            numeric_features = preprocessor_data['numeric_features']
            categorical_features = preprocessor_data['categorical_features']
            
            # Transform input
            input_processed = preprocessor.transform(input_data[numeric_features + categorical_features])
            
            # Predictions
            freight_cost = regressor.predict(input_processed)[0]
            risk_proba = classifier.predict_proba(input_processed)[0]
            risk_label = "HIGH RISK" if risk_proba[1] > 0.5 else "LOW RISK"
            
            # Display results
            st.divider()
            st.subheader("Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Predicted Freight Cost", f"${freight_cost:,.2f}")
            
            with res_col2:
                st.metric("Risk Probability", f"{risk_proba[1]*100:.1f}%")
            
            with res_col3:
                color = "🔴" if risk_proba[1] > 0.5 else "🟢"
                st.metric("Risk Assessment", f"{color} {risk_label}")
            
            if risk_proba[1] > 0.5:
                st.warning("⚠️ This invoice is flagged as high risk. Manual review recommended.")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def show_model_insights(regressor, classifier, df, preprocessor_data):
    """Display model insights and SHAP explanations."""
    
    st.header("Model Insights")
    
    # Load metrics
    import json
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Model (Freight Cost)")
        reg_metrics = metrics['regression']
        st.metric("R² Score", f"{reg_metrics['r2']:.4f}")
        st.metric("MAE", f"${reg_metrics['mae']:.2f}")
        st.metric("RMSE", f"${reg_metrics['rmse']:.2f}")
    
    with col2:
        st.subheader("Classification Model (Risk Detection)")
        clf_metrics = metrics['classification']
        st.metric("Accuracy", f"{clf_metrics['accuracy']:.4f}")
        st.metric("Precision", f"{clf_metrics['precision']:.4f}")
        st.metric("F1-Score", f"{clf_metrics['f1']:.4f}")
    
    st.divider()
    
    # Feature importance
    st.subheader("Feature Importance (Regression)")
    
    feature_importance = pd.DataFrame({
        'Feature': preprocessor_data['feature_names'],
        'Importance': regressor.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features for Freight Cost Prediction')
    plt.tight_layout()
    st.pyplot(fig)
    
    # SHAP summary
    st.subheader("SHAP Feature Impact")
    
    X_sample = df.drop(columns=['freight_cost', 'is_risky', 'risk_score', 'invoice_id', 'invoice_date']).head(100)
    X_sample = X_sample.replace([np.inf, -np.inf], np.nan)
    
    try:
        preprocessor = preprocessor_data['preprocessor']
        X_processed = preprocessor.transform(X_sample)
        
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_processed)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, 
                         feature_names=preprocessor_data['feature_names'],
                         show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {str(e)}")

def show_data_explorer(df):
    """Interactive data exploration."""
    
    st.header("Data Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vendor_filter = st.multiselect("Filter by Vendor", df['vendor_name'].unique())
    
    with col2:
        carrier_filter = st.multiselect("Filter by Carrier", df['carrier_type'].unique())
    
    with col3:
        risk_filter = st.selectbox("Filter by Risk", ["All", "Low Risk", "High Risk"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if vendor_filter:
        filtered_df = filtered_df[filtered_df['vendor_name'].isin(vendor_filter)]
    
    if carrier_filter:
        filtered_df = filtered_df[filtered_df['carrier_type'].isin(carrier_filter)]
    
    if risk_filter == "Low Risk":
        filtered_df = filtered_df[filtered_df['is_risky'] == 0]
    elif risk_filter == "High Risk":
        filtered_df = filtered_df[filtered_df['is_risky'] == 1]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_invoices.csv",
        mime="text/csv"
    )

if __name__ == '__main__':
    main()
