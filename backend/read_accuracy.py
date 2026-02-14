try:
    with open("accuracy_debug.log", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read accuracy log: {e}")
