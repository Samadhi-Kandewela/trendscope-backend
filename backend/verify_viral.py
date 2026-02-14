import sys
import logging
from app import create_app

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

def test_viral_prediction():
    print("--- Verifying Viral Velocity Model ---")
    
    with app.test_client() as client:
        # 1. Test "High Potential" Scenario
        # 50k views in 2 hours -> Should be high
        print("\nTesting Scenario 1: 50k views, 5k likes, 2 hours")
        resp = client.get("/api/analytics/viral-potential?views=50000&likes=5000&comments=500&hours_since_upload=2")
        
        msg = ""
        msg += f"Status: {resp.status_code}\n"
        msg += f"Body: {resp.get_data(as_text=True)}\n"
        print(msg)
        
        # 2. Test "Low Potential" Scenario
        print("\nTesting Scenario 2: 100 views, 2 likes, 48 hours")
        resp = client.get("/api/analytics/viral-potential?views=100&likes=2&comments=0&hours_since_upload=48")
        msg += "\n--- Scenario 2 ---\n"
        msg += f"Body: {resp.get_data(as_text=True)}\n"
        print(msg)
        
        with open("viral_verify_out.txt", "w") as f:
            f.write(msg)

if __name__ == "__main__":
    test_viral_prediction()
