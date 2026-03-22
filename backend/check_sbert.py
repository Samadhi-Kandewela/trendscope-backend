
print("Start S-BERT check", flush=True)
try:
    from sentence_transformers import SentenceTransformer
    print("S-BERT imported", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("S-BERT Model loaded", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
