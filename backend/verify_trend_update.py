import os
import sys
from datetime import datetime, timedelta

# Avoid import errors
sys.path.append(os.getcwd())

from app import create_app, db

app = create_app()
with app.app_context():
    from app.services.trend_service import get_trend_strategy
    # from app.ml.validation import ValidationService
    
    print("--- 1. Testing Trend Service (New Metrics) ---")
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()
    
    # Test without User ID (Baseline)
    try:
        strategy = get_trend_strategy("US", start, end, model_type="baseline")
        
        if strategy:
            print(f"Success! Core Trend: {strategy.get('coreTrend')}")
            print(f"Data Volume: {strategy.get('dataVolume')}")
            print(f"Relevance Score: {strategy.get('relevanceScore')}")
            print(f"Prediction Confidence: {strategy.get('predictionConfidence')}")
            print(f"Sentiment: {strategy.get('sentiment')}")
            
            metrics = strategy.get('trendMetrics', {})
            first_kw = list(metrics.keys())[0] if metrics else "None"
            print(f"Trend Metrics (sample '{first_kw}'): {metrics.get(first_kw)}")
            
            if strategy.get('suggestedVideos'):
                v = strategy['suggestedVideos'][0]
                print(f"Suggested Video 1 Similarity: {v.get('similarityScore')}")
                print(f"Sample Method: {v.get('selectionMethod')}")
        else:
            print("Strategy generation returned None (possibly due to fallback).")
            
    except Exception as e:
        print(f"Error testing Trend Service: {e}")
        import traceback
        traceback.print_exc()

    '''
    print("\n--- 2. Testing Validation Service (Real Metrics) ---")
    validator = ValidationService()
    try:
        # Run Backtest (Quick)
        print("Running Keyword Hit Rate Backtest...")
        backtest = validator.run_historical_backtest("US")
        if "error" in backtest:
             print(f"Backtest Error: {backtest['error']}")
        else:
             print("Backtest Result (Details):")
             print(backtest.get("details", {}).get("metrics"))
        
        # Run Clustering Quality (Sampled)
        print("\nRunning Clustering Quality Check (Intrinsic/External)...")
        quality = validator.run_clustering_quality_check()
        if "error" in quality:
             print(f"Quality Check Error: {quality['error']}")
        else:
             print("Quality Check Result (Details):")
             print(quality.get("details", {}).get("group_a_intrinsic"))
             print(quality.get("details", {}).get("group_b_external_genre_alignment"))
        
    except Exception as e:
        print(f"Validation Error: {e}")
        import traceback
        traceback.print_exc()
    '''
