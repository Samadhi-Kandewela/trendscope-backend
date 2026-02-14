import sys
import logging
from app import create_app, db
from app.models.community import Discussion

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

def test_strategist_ranking():
    print("--- Verifying Top Strategists ---")
    
    with app.app_context():
        # 1. Clear existing
        try:
            db.session.query(Discussion).delete()
            db.session.commit()
            print("Cleared existing discussions.")
        except Exception as e:
            db.session.rollback()
            print(f"Error clearing: {e}")
            return

        # 2. Create posts
        posts_data = [
            {"title": "Post A1", "author": "UserA", "votes": 10},
            {"title": "Post A2", "author": "UserA", "votes": 5},  # UserA Total: 15
            {"title": "Post B1", "author": "UserB", "votes": 20}, # UserB Total: 20
            {"title": "Post C1", "author": "UserC", "votes": 0},
        ]
        
        for p in posts_data:
            post = Discussion(title=p['title'], author_username=p['author'], upvotes=p['votes'])
            db.session.add(post)
        
        db.session.commit()
        print("Created test data: UserA(15), UserB(20), UserC(0)")

    # 3. Test API via Client (outside app context, but client creates its own request context)
    with app.test_client() as client:
        print("\nFetching Top Strategists...")
        try:
            resp = client.get("/api/community/strategists")
            if resp.status_code == 200:
                ranking = resp.get_json()
                print(f"Ranking: {ranking}")
                
                if len(ranking) >= 2:
                    # UserB should be first (20), UserA second (15)
                    if ranking[0]['username'] == "UserB" and ranking[1]['username'] == "UserA":
                         print("SUCCESS: UserB is #1, UserA is #2.")
                    else:
                         print("FAILURE: Incorrect ranking order.")
                else:
                     print("FAILURE: Not enough results returned.")
            else:
                print(f"Failed to fetch: {resp.status_code}")
                print(resp.get_data(as_text=True))
        except Exception as e:
            print(f"Error during request: {e}")

if __name__ == "__main__":
    test_strategist_ranking()
