import sys
import logging
from app import create_app, db
from app.models.community import Discussion

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import os
os.environ["WERKZEUG_RUN_MAIN"] = "true" # Disable scheduler

app = create_app()

def test_tags_ranking():
    print("--- Verifying Trending Tags ---")
    
    with app.app_context():
        # 1. Clear existing
        try:
            db.session.query(Discussion).delete()
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error clearing: {e}")

        # 2. Create posts
        posts_data = [
            {"title": "P1", "tags": ["gaming", "strategy"]},
            {"title": "P2", "tags": ["gaming"]},
            {"title": "P3", "tags": ["gaming", "strategy", "productivity"]},
            {"title": "P4", "tags": []}, 
        ]
        
        for p in posts_data:
            post = Discussion(title=p['title'], tags=p['tags'])
            db.session.add(post)
        db.session.commit()

    # 3. Test API
    with app.test_client() as client:
        print("\nFetching Top Tags...")
        try:
            resp = client.get("/api/community/tags")
            with open("tags_verify_out.txt", "w") as f:
                f.write(f"Status: {resp.status_code}\n")
                f.write(f"Body: {resp.get_data(as_text=True)}\n")
            
            print(f"Body: {resp.get_data(as_text=True)}")
        except Exception as e:
            with open("tags_verify_out.txt", "w") as f:
                f.write(f"Error: {e}")
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error during request: {e}")

if __name__ == "__main__":
    test_tags_ranking()
