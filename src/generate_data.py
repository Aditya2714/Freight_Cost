import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

VENDORS = ['FastShip Logistics', 'Global Freight Co', 'Quick Deliveries Inc', 
           'Prime Transport', 'Express Cargo', 'Swift Logistics', 
           'Reliable Shipping', 'Premier Freight', 'Elite Transport', 'Summit Logistics']

PRODUCTS = ['Electronics', 'Clothing', 'Food & Beverage', 'Machinery', 
            'Chemicals', 'Textiles', 'Automotive Parts', 'Pharmaceuticals', 
            'Building Materials', 'Agricultural Products']

ORIGINS = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
           'Shanghai', 'Mumbai', 'London', 'Tokyo', 'Dubai']

DESTINATIONS = ['Seattle', 'Boston', 'Denver', 'Miami', 'Atlanta',
                'San Francisco', 'Portland', 'Dallas', 'Detroit', 'Nashville']

CARRIER_TYPES = ['Air', 'Sea', 'Rail', 'Truck']
RISK_LABELS = ['Low', 'Medium', 'High']

def generate_invoice_data(n_records=5000):
    """Generate synthetic invoice dataset for freight cost prediction."""
    
    data = {
        'invoice_id': [f'INV-{1000+i}' for i in range(n_records)],
        'invoice_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) 
                        for _ in range(n_records)],
        'vendor_name': np.random.choice(VENDORS, n_records),
        'product_category': np.random.choice(PRODUCTS, n_records),
        'origin_city': np.random.choice(ORIGINS, n_records),
        'destination_city': np.random.choice(DESTINATIONS, n_records),
        'carrier_type': np.random.choice(CARRIER_TYPES, n_records),
        'weight_kg': np.round(np.random.uniform(0.5, 5000, n_records), 2),
        'volume_m3': np.round(np.random.uniform(0.01, 50, n_records), 2),
        'quantity': np.random.randint(1, 1000, n_records),
        'distance_km': np.round(np.random.uniform(100, 15000, n_records), 0),
        'base_rate_per_km': np.round(np.random.uniform(0.5, 5.0, n_records), 2),
        'fuel_surcharge_pct': np.round(np.random.uniform(5, 25, n_records), 1),
        'insurance_value': np.round(np.random.uniform(100, 500000, n_records), 2),
        'handling_fee': np.round(np.random.uniform(10, 500, n_records), 2),
        'delivery_days': np.random.randint(1, 30, n_records),
        'is_expedited': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
        'customs_clearance': np.random.choice([0, 1], n_records, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate freight cost (target for regression)
    base_freight = df['distance_km'] * df['base_rate_per_km'] * (df['weight_kg'] / 100)
    fuel_surcharge = base_freight * (df['fuel_surcharge_pct'] / 100)
    volume_charge = df['volume_m3'] * 100
    expedited_premium = df['is_expedited'] * base_freight * 0.3
    customs_fee = df['customs_clearance'] * 250
    
    df['freight_cost'] = np.round(
        base_freight + fuel_surcharge + volume_charge + 
        expedited_premium + customs_fee + df['handling_fee'] + 
        np.random.normal(0, 50, n_records), 2
    )
    df['freight_cost'] = df['freight_cost'].clip(lower=50)
    
    # Calculate risk score and label (target for classification)
    risk_score = np.zeros(n_records)
    
    # High value + low quantity = suspicious
    risk_score += (df['insurance_value'] > 100000) * 0.3
    risk_score += (df['quantity'] < 10) * 0.2
    
    # Unusual freight cost relative to distance
    cost_per_km = df['freight_cost'] / df['distance_km']
    risk_score += (cost_per_km > cost_per_km.quantile(0.9)) * 0.4
    
    # New vendor patterns
    vendor_avg_cost = df.groupby('vendor_name')['freight_cost'].transform('mean')
    risk_score += ((df['freight_cost'] - vendor_avg_cost).abs() > vendor_avg_cost * 0.5) * 0.3
    
    # Add some noise
    risk_score += np.random.uniform(0, 0.2, n_records)
    
    df['risk_score'] = np.round(risk_score, 3)
    df['is_risky'] = (df['risk_score'] > 0.5).astype(int)
    
    return df

def main():
    print("Generating synthetic invoice dataset...")
    df = generate_invoice_data(5000)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_invoices.csv', index=False)
    
    print(f"Dataset generated: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFreight Cost Statistics:")
    print(df['freight_cost'].describe())
    print(f"\nRisk Distribution:")
    print(df['is_risky'].value_counts())
    print(f"\nSaved to: data/raw_invoices.csv")

if __name__ == '__main__':
    main()
