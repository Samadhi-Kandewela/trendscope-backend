import sys
import os

# Add backend to path to import app
sys.path.append(os.getcwd())

from app import create_app
from app.services.scheduler import run_ingestion

def trigger_ingestion():
    app = create_app()
    with app.app_context():
        print("--- Triggering Live Ingestion ---")
        try:
            # Reuses the same logic as the scheduler
            # Takes 'app' as argument to create its own context or use existing
            run_ingestion(app)
            print("SUCCESS: Ingestion function completed.")
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    trigger_ingestion()
