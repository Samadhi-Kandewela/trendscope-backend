
import logging
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)

from app import create_app
from app.services.trend_service import get_trend_strategy

def verify_hybrid_model():
    app = create_app()
    with app.app_context():
        print("\n--- Verifying Hybrid Model (Enhanced TF-IDF + K-Means) ---\n")
        
        # Test Parameters
        region = "US"
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)
        
        print(f"Region: {region}")
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        
        # Call Strategy Service
        print("\n> Calling get_trend_strategy(model_type='hybrid')...")
        strategy = get_trend_strategy(
            region=region,
            start_date=start_date,
            end_date=end_date,
            model_type="hybrid"
        )
        
        if strategy:
            print("\n✅ SUCCESS: Hybrid Model returned a strategy.")
            print("-" * 40)
            print(f"Core Genre: {strategy.get('coreGenre')}")
            print(f"Core Trend: {strategy.get('coreTrend')}")
            print(f"Keywords:   {strategy.get('topKeywords')}")
            print("-" * 40)
            
            # Check if keywords look "rich" (subjective, but S-BERT usually gives diverse terms)
            keywords = strategy.get('topKeywords', [])
            if len(keywords) > 3:
                print("Keyword count looks good.")
            else:
                print("WARNING: Few keywords returned.")
        else:
            print("\n❌ FAILURE: Hybrid Model returned None.")

if __name__ == "__main__":
    verify_hybrid_model()
