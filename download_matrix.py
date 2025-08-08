import requests
from pathlib import Path

# Create data folder
Path("data").mkdir(exist_ok=True)

# Download your matrix
url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
response = requests.get(url)

# Save it
with open("data/distance_matrix.csv", "w") as f:
    f.write(response.text)

print("✅ Matrix saved to data/distance_matrix.csv")
