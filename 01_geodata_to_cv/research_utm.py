import json

with open("/Users/Di/Documents/Новая папка/Geodata/004_ДЕМИДОВКА/UTM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)