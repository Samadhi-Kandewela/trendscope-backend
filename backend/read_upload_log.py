try:
    with open("upload_debug.log", encoding="utf-16", errors="ignore") as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(line.strip())
except Exception as e:
    print(e)
