import logging
import sys
from app import create_app
from app.ml.validation import ModelValidator
from app.services.trend_service import get_trend_strategy
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

with app.app_context():
    print("--- Verifying Accuracy Framework ---")
    
    validator = ModelValidator()
    
    print("\n1. Running Historical Backtest (Simulation)...")
    backtest_res = validator.run_historical_backtest()
    print(f"Backtest Score: {backtest_res.get('accuracy_score')}")
    
    print("\n2. Running Live Accuracy Check...")
    live_res = validator.run_live_accuracy_check()
    print(f"Live Score: {live_res.get('accuracy_score')}")
    print(f"Details: {live_res.get('details')}")
    
    print("\n3. Testing Trend Strategy with Live Boost...")
    # This will implicitly test the code changes in trend_service.py
    start = datetime.utcnow()
    end = start + timedelta(days=30)
    strategy = get_trend_strategy("Global", start, end)
    
    if strategy:
        print(f"Strategy Generated: {strategy['coreTrend']}")
        print(f"Source: {strategy.get('strategySource')}")
        # Check if live videos contributed
        # We can't easily check "contribution" from outside, but we check if it runs.
    else:
        print("Strategy Generation Failed.")
        
    print("\nVerification Complete.")
