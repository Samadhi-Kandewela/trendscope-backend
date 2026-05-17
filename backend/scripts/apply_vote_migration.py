from app import create_app, db
from app.models.community import DiscussionVote

app = create_app()

with app.app_context():
    print("Applying migration for DiscussionVote table...")
    try:
        # Create table logic
        # Use create_all for safer selective creation if bound to specific bind
        # or just table.create(db.engine)
        DiscussionVote.__table__.create(db.engine)
        print("Success: 'discussion_votes' table created.")
    except Exception as e:
        print(f"Error creating table: {e}")
