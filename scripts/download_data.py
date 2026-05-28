import urllib.request
import zipfile
import os

url = 'https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip'
dest_dir = os.path.expanduser('~/.causalnerve/data')
os.makedirs(dest_dir, exist_ok=True)
zip_path = os.path.join(dest_dir, 'CMAPSSData.zip')

print("Downloading...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    content = response.read()
    with open(zip_path, 'wb') as f:
        f.write(content)

print(f"Downloaded size: {len(content)} bytes")

try:
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print("Done!")
except Exception as e:
    print("Extraction failed:", e)
    with open(zip_path, 'r', errors='ignore') as f:
        print("File starts with:", f.read(200))
