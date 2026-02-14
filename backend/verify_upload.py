import sys
import os
import io
import logging
from app import create_app

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Hack: Disable scheduler for this test to reduce noise
os.environ["WERKZEUG_RUN_MAIN"] = "true" 

app = create_app()

def test_upload_flow():
    with app.test_client() as client:
        print("--- Verifying File Upload ---")
        
        # 1. Create a dummy image file (bytes)
        data = {
            'file': (io.BytesIO(b"dummy image data"), 'test_image.jpg')
        }
        
        # 2. Upload
        print("\n1. Uploading 'test_image.jpg'...")
        # content_type='multipart/form-data' is handled automatically by client.post when data has file
        resp = client.post("/api/upload/", data=data, content_type='multipart/form-data')
        
        if resp.status_code == 201:
            res_json = resp.get_json()
            with open("upload_result.txt", "w") as f:
                f.write(f"SUCCESS: {res_json['url']}")
            print(f"Success! File URL: {res_json['url']}")
        else:
            with open("upload_result.txt", "w") as f:
                f.write(f"FAILURE: {resp.status_code}")
            print(f"Failed to upload: {resp.status_code}")

if __name__ == "__main__":
    test_upload_flow()
