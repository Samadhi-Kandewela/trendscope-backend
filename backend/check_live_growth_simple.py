
from sqlalchemy import create_engine, text

# Hardcoded for simplicity/speed since I know it from .env
DB_URL = "postgresql://postgres:samadhi@localhost:5432/trendscope_correct"

def check_growth():
    print("Connecting to DB (Directly)...")
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            print("Connected.")
            
            # 1. Total Live Videos
            print("Querying total count...")
            result = conn.execute(text("SELECT count(*) FROM videos WHERE source_dataset = 'live_api'"))
            total = result.scalar()
            print(f"Total Live Videos: {total}")
            
            # 2. Daily Breakdown
            print("Querying daily breakdown...")
            # Using date_trunc or casting to date for postgres
            query = text("""
                SELECT date(trending_date) as day, count(*) as cnt 
                FROM videos 
                WHERE source_dataset = 'live_api' 
                GROUP BY day 
                ORDER BY day DESC
                LIMIT 10
            """)
            rows = conn.execute(query).fetchall()
            
            if not rows:
                print("\nNo daily data found. Checking if trending_date is NULL...")
                null_check = conn.execute(text("SELECT count(*) FROM videos WHERE source_dataset = 'live_api' AND trending_date IS NULL")).scalar()
                print(f"  Videos with NULL trending_date: {null_check}")
            else:
                print("\nRecent Daily Ingestion (Last 10 Days):")
                for row in rows:
                    print(f"  {row[0]}: {row[1]} videos")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_growth()
