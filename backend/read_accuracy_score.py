from collections import deque
try:
    with open("accuracy_fix_debug.log", encoding="utf-16", errors="ignore") as f:
        last_lines = deque(f, maxlen=20)
        for line in last_lines:
            print(line.strip())
except Exception as e:
    print(f"Error reading log: {e}")
