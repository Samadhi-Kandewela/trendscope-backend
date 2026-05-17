from app import create_app, db
from app.models.community import Discussion, DiscussionComment

app = create_app()

with app.app_context():
    print("Clearing all discussions and comments...")
    try:
        # Delete using SQL objects
        num_comments = db.session.query(DiscussionComment).delete()
        num_discussions = db.session.query(Discussion).delete()
        db.session.commit()
        print(f"Deleted {num_comments} comments and {num_discussions} discussions.")
    except Exception as e:
        print(f"Error clearing tables: {e}")
        db.session.rollback()
