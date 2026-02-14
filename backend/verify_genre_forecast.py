import sys
import logging
from datetime import datetime, timedelta
from app import create_app
from app.ml.genre_forecast import get_genre_forecast_distribution

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

with app.app_context():
    print("--- Verifying Hybrid Genre Forecast ---")
    
    # 1. Test Default (Next 30 Days)
    print("\n1. Testing Default (Next 30 Days)...")
    defaults = get_genre_forecast_distribution("US")
    print(f"Top Genre: {defaults[0]['name']} ({defaults[0]['percentage']}%)")
    
    # 2. Test Future Date Range (April 2026)
    print("\n2. Testing Future Range (April 2026)...")
    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 30)
    future = get_genre_forecast_distribution("US", start_date=start, end_date=end)
    print(f"Top Genre: {future[0]['name']} ({future[0]['percentage']}%)")
    
    # 3. Test Live Data Fusion
    # We can't easily assert the exact percentage without knowing the DB state,
    # but we can check if it runs without error.
    print("\n3. Testing Live Data Fusion Logic...")
    # This implicitly tests _get_live_genre_distribution
    hybrid = get_genre_forecast_distribution("Global") 
    print(f"Global Top: {hybrid[0]['name']} ({hybrid[0]['percentage']}%)")
    
    print("\nVerification Complete.")
