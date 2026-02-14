from app import create_app
from app.extensions import db
from app.services.data_manager import DataMerger

app = create_app()

with app.app_context():
    print("--- Database Composition ---")
    merger = DataMerger(db.session)
    stats = merger.get_dataset_stats()
    
    for source, count in stats.items():
        if source == "total":
            continue
        print(f"Dataset '{source}': {count} videos")
        
    print(f"----------------------------")
    print(f"Total Videos: {stats['total']}")
