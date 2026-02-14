from app import create_app, db
from app.models.community import Discussion, DiscussionComment

app = create_app()

with app.app_context():
    print("Creating 'discussions' and 'discussion_comments' tables...")
    try:
        db.create_all() # This is safe; it only creates tables that don't exist
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
