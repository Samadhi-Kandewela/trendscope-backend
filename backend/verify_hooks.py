import sys
import logging
from datetime import datetime
from app import create_app
from app.services.trend_service import get_trend_strategy

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

with app.app_context():
    print("--- Verifying Marketing Hooks ---")
    
    # 1. Generate Strategy for US, next month
    start_date = datetime(2026, 3, 1)
    end_date = datetime(2026, 3, 31)
    
    print(f"Requesting strategy for {start_date.date()} to {end_date.date()}...")
    
    result = get_trend_strategy("US", start_date, end_date)
    
    if result:
        print("\n[Detailed Strategy]")
        print(result.get("detailedStrategy")[:200] + "...")
        
        print("\n[Marketing Insights (Hooks)]")
        hooks = result.get("marketingInsights", [])
        for h in hooks:
            print(f"- {h}")
            
        print(f"\nSource: {result.get('strategySource')}")
    else:
        print("No strategy returned (insufficient data?).")
