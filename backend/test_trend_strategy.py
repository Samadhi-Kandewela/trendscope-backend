import logging
import sys
from datetime import datetime, timedelta
from app import create_app
from app.services.trend_service import get_trend_strategy

# output to file to avoid encoding issues on console
logging.basicConfig(filename='strategy_test_debug.log', level=logging.INFO)

app = create_app()

with app.app_context():
    # Test for a future period (e.g. next month)
    start = datetime.now() + timedelta(days=30)
    end = start + timedelta(days=60)
    
    print(f"Testing Strategy for: {start.date()} to {end.date()}")
    
    # We use "US" and "Lifestyle" (or general)
    # The function takes region, start, end. It finds its own genre from the data.
    # Let's try region="US"
    
    result = get_trend_strategy("US", start, end)
    
    if result:
        print("\n--- RESULT ---")
        print(f"Core Trend Label: {result['coreTrend']}")
        print(f"Strategy Source: {result.get('strategySource')}")
        print("\n--- Detailed Strategy ---")
        print(result['detailedStrategy'])
        
        # Check for year leakage in label
        if "2018" in result['coreTrend'] or "2020" in result['coreTrend']:
            print("\n[FAIL] Found past year in Core Trend Label!")
        else:
            print("\n[PASS] Core Trend Label looks clean.")
            
        # Check for Year in strategy (it should be 2026 or generic)
        if "2018" in result['detailedStrategy']:
             print("[WARNING] Strategy text mentions 2018. Check context.")
        else:
             print("[PASS] Strategy text avoids 2018.")
             
    else:
        print("No strategy returned (maybe insufficient data).")
