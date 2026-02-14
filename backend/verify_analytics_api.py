import sys
import logging
from app import create_app
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

def test_endpoint():
    with app.test_client() as client:
        print("--- Verifying Analytics API Endpoint ---")
        
        # 1. Test Default (No dates)
        print("\n1. Requesting Default (Next 30 Days)...")
        resp_default = client.get("/api/analytics/genres/near-term?region=US")
        if resp_default.status_code == 200:
            data = resp_default.get_json()
            print(f"Summary: {data.get('summary')}")
            print(f"Top Genre: {data['genres'][0]['name']} ({data['genres'][0]['percentage']}%)")
        else:
            print(f"Error: {resp_default.status_code}")

        # 2. Test Specific Future Dates
        print("\n2. Requesting Future Date Range (April 2026)...")
        # Construct URL with query params
        start_date = "2026-04-01"
        end_date = "2026-04-30"
        url = f"/api/analytics/genres/near-term?region=US&startDate={start_date}&endDate={end_date}"
        
        resp_future = client.get(url)
        if resp_future.status_code == 200:
            data = resp_future.get_json()
            print(f"Summary: {data.get('summary')}")
            print(f"Top Genre: {data['genres'][0]['name']} ({data['genres'][0]['percentage']}%)")
        else:
            print(f"Error: {resp_future.status_code}")
            print(resp_future.get_data(as_text=True))
            
if __name__ == "__main__":
    test_endpoint()
