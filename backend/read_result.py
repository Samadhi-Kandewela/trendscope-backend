try:
    with open("sentiment_result.txt", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read result file: {e}")
