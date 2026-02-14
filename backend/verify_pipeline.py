from app import create_app
from app.services.scheduler import run_ingestion

import logging
import sys
# force utf-8 file output
logging.basicConfig(filename='pipeline_debug.log', filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = create_app()

print("Starting manual pipeline verification...")
try:
    # Run the ingestion logic directly
    run_ingestion(app)
    print("Verification SUCCESS: Pipeline ran without errors.")
except Exception as e:
    print(f"Verification FAILED: {e}")
