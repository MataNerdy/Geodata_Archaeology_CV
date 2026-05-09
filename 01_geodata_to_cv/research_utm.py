import json

with open("/Volumes/Lexar/Датасет/004_ДЕМИДОВКА/UTM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)