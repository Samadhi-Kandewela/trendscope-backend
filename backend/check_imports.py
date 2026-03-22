
import sys
import time

def log(msg):
    print(msg, flush=True)

log("Starting imports check")
time.sleep(0.5)

log("Importing os...")
import os

log("Importing joblib...")
import joblib

log("Importing sqlalchemy...")
try:
    from sqlalchemy import create_engine
    log("SQLAlchemy imported.")
except ImportError:
    log("SQLAlchemy ImportError.")
except Exception as e:
    log(f"SQLAlchemy Exception: {e}")

log("Importing sklearn.feature_extraction...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    log("TfidfVectorizer imported.")
except ImportError:
    log("sklearn ImportError.")
except Exception as e:
    log(f"sklearn Exception: {e}")

log("Importing sklearn.cluster...")
try:
    from sklearn.cluster import KMeans
    log("KMeans imported.")
except ImportError:
    log("sklearn ImportError.")
except Exception as e:
    log(f"sklearn Exception: {e}")

log("Done all imports")
