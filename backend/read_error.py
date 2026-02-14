try:
    with open("sentiment_error.txt", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read error file: {e}")
