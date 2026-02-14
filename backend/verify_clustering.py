import sys
import logging
from app import create_app, db
from app.ml.validation import ModelValidator
from app.models.accuracy import AccuracyLog

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

def test_clustering_check():
    print("--- Verifying Clustering Quality Check ---")
    
    with app.app_context():
        validator = ModelValidator()
        
        print("Running Clustering Check...")
        try:
            result = validator.run_clustering_quality_check()
            print(f"Result: {result}")
            
            if "error" in result:
                print("FAILURE: Validation returned error.")
                return

            # Verify Log
            log = db.session.query(AccuracyLog).filter_by(
                log_type='clustering_quality'
            ).order_by(AccuracyLog.log_date.desc()).first()
            
            with open("clustering_verify_out.txt", "w") as f:
                if log:
                    msg = f"SUCCESS: Log saved. Score: {log.accuracy_score}\nDetails: {log.details}"
                    print(msg)
                    f.write(msg)
                else:
                    msg = "FAILURE: No log found in DB."
                    print(msg)
                    f.write(msg)
                
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            with open("clustering_verify_out.txt", "w") as f:
                f.write(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_clustering_check()
