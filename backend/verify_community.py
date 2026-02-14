import sys
import logging
from app import create_app

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

def test_community_flow():
    with app.test_client() as client:
        print("--- Verifying Community Features ---")
        
        # 1. Create a "Instagram-style" Discussion
        print("\n1. Creating a Visual Discussion Post...")
        payload = {
            "title": "Strategy for March: Cozy Gaming",
            "body": "I think the 'Cozy' trend is evolving into 'Cozy Productivity'. See chart attached.",
            "tags": ["gaming", "strategy", "march"],
            "media_url": "https://example.com/chart.png",
            "linked_context": {"region": "US", "months": [3, 4]},
            "author_username": "SarahCreator"
        }
        resp = client.post("/api/community/discussions", json=payload)
        
        if resp.status_code == 201:
            post = resp.get_json()
            post_id = post['id']
            print(f"Success! Created Post ID {post_id}: {post['title']}")
            print(f"Media URL: {post.get('media_url')}")
        else:
            print(f"Failed to create post: {resp.status_code}")
            print(resp.get_data(as_text=True))
            return

        # 2. Add a Comment
        print("\n2. Adding a Comment...")
        comment_payload = {
            "body": "Totally agree! I've seen this in my analytics too.",
            "author_username": "TechGuru"
        }
        resp_comment = client.post(f"/api/community/discussions/{post_id}/comments", json=comment_payload)
        
        if resp_comment.status_code == 201:
            print("Success! Comment added.")
        else:
            print(f"Failed to add comment: {resp_comment.status_code}")

        # 3. Upvote
        print("\n3. Upvoting the Post...")
        resp_vote = client.post(f"/api/community/discussions/{post_id}/vote", json={"type": "up"})
        
        if resp_vote.status_code == 200:
            print(f"Success! New Upvote Count: {resp_vote.get_json()['upvotes']}")
        else:
            print(f"Failed to vote: {resp_vote.status_code}")

        # 4. Fetch Feed
        print("\n4. Fetching 'Recent' Feed...")
        resp_feed = client.get("/api/community/discussions?sort=recent")
        if resp_feed.status_code == 200:
            feed = resp_feed.get_json()
            print(f"Feed contains {len(feed)} posts.")
            print(f"Top Post: {feed[0]['title']} (Upvotes: {feed[0]['upvotes']})")
        else:
            print(f"Failed to fetch feed: {resp_feed.status_code}")

if __name__ == "__main__":
    test_community_flow()
