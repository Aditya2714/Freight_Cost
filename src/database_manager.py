import duckdb
import pandas as pd
import os

class DatabaseManager:
    """SQL-first feature engineering using DuckDB."""
    
    def __init__(self, db_path=':memory:'):
        self.conn = duckdb.connect(db_path)
    
    def load_data(self, csv_path):
        """Load CSV data into DuckDB."""
        df = pd.read_csv(csv_path)
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        self.conn.execute("CREATE TABLE invoices AS SELECT * FROM df")
        return df
    
    def create_features(self):
        """Create SQL-based features using CTEs and window functions."""
        
        feature_query = """
        WITH vendor_stats AS (
            SELECT 
                vendor_name,
                AVG(freight_cost) as avg_vendor_cost,
                STDDEV(freight_cost) as std_vendor_cost,
                COUNT(*) as vendor_transaction_count,
                AVG(delivery_days) as avg_delivery_days
            FROM invoices
            GROUP BY vendor_name
        ),
        
        route_stats AS (
            SELECT 
                origin_city,
                destination_city,
                AVG(freight_cost) as avg_route_cost,
                AVG(distance_km) as avg_route_distance,
                COUNT(*) as route_count
            FROM invoices
            GROUP BY origin_city, destination_city
        ),
        
        product_stats AS (
            SELECT 
                product_category,
                AVG(weight_kg) as avg_product_weight,
                AVG(volume_m3) as avg_product_volume,
                AVG(freight_cost) as avg_product_freight
            FROM invoices
            GROUP BY product_category
        ),
        
        cost_per_unit AS (
            SELECT 
                *,
                freight_cost / NULLIF(quantity, 0) as cost_per_unit,
                freight_cost / NULLIF(distance_km, 0) as cost_per_km,
                freight_cost / NULLIF(weight_kg, 0) as cost_per_kg,
                weight_kg / NULLIF(volume_m3, 0) as density
            FROM invoices
        ),
        
        monthly_stats AS (
            SELECT 
                *,
                EXTRACT(MONTH FROM invoice_date) as invoice_month,
                EXTRACT(DAY FROM invoice_date) as invoice_day,
                EXTRACT(DOW FROM invoice_date) as invoice_day_of_week
            FROM cost_per_unit
        ),
        
        enriched AS (
            SELECT 
                m.*,
                v.avg_vendor_cost,
                v.std_vendor_cost,
                v.vendor_transaction_count,
                v.avg_delivery_days as vendor_avg_delivery,
                r.avg_route_cost,
                r.avg_route_distance,
                r.route_count,
                p.avg_product_weight,
                p.avg_product_volume,
                p.avg_product_freight,
                -- Deviation from vendor average
                (m.freight_cost - v.avg_vendor_cost) / NULLIF(v.std_vendor_cost, 0) as vendor_cost_zscore,
                -- Deviation from route average
                (m.freight_cost - r.avg_route_cost) / NULLIF(r.avg_route_cost, 0) as route_cost_deviation,
                -- Risk indicators
                CASE 
                    WHEN m.insurance_value > 100000 THEN 1 ELSE 0 
                END as high_value_flag,
                CASE 
                    WHEN m.quantity < 10 THEN 1 ELSE 0 
                END as low_quantity_flag,
                CASE 
                    WHEN m.freight_cost > v.avg_vendor_cost * 1.5 THEN 1 ELSE 0 
                END as above_avg_cost_flag
            FROM monthly_stats m
            LEFT JOIN vendor_stats v ON m.vendor_name = v.vendor_name
            LEFT JOIN route_stats r ON m.origin_city = r.origin_city 
                AND m.destination_city = r.destination_city
            LEFT JOIN product_stats p ON m.product_category = p.product_category
        )
        
        SELECT * FROM enriched
        """
        
        result = self.conn.execute(feature_query).fetchdf()
        return result
    
    def get_summary_stats(self):
        """Get summary statistics from the database."""
        query = """
        SELECT 
            COUNT(*) as total_records,
            AVG(freight_cost) as avg_freight_cost,
            STDDEV(freight_cost) as std_freight_cost,
            MIN(freight_cost) as min_freight_cost,
            MAX(freight_cost) as max_freight_cost,
            AVG(distance_km) as avg_distance,
            AVG(weight_kg) as avg_weight,
            SUM(CASE WHEN is_risky = 1 THEN 1 ELSE 0 END) as risky_count
        FROM invoices
        """
        return self.conn.execute(query).fetchdf()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

def main():
    """Run feature engineering pipeline."""
    db = DatabaseManager()
    
    print("Loading raw data...")
    db.load_data('data/raw_invoices.csv')
    
    print("Creating SQL features...")
    featured_df = db.create_features()
    
    print("Saving featured data...")
    featured_df.to_csv('data/featured_invoices.csv', index=False)
    
    print("\nSummary Statistics:")
    print(db.get_summary_stats())
    
    print(f"\nFeatured dataset shape: {featured_df.shape}")
    print(f"New columns: {[col for col in featured_df.columns if col not in pd.read_csv('data/raw_invoices.csv').columns]}")
    
    db.close()

if __name__ == '__main__':
    main()
