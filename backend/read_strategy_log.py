try:
    with open("test_strategy.log", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read log: {e}")
