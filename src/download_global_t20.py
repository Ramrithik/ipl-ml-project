import urllib.request
import zipfile
import os
import ssl

url = 'https://cricsheet.org/downloads/t20s_csv2.zip'
zip_path = r'C:\mydesk\archive\t20s_csv2.zip'
extract_dir = r'C:\mydesk\archive\t20s'

print('Bypassing SSL and downloading ~25MB zip from Cricsheet...')
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, 'wb') as out_file:
    data = response.read()
    out_file.write(data)

print('Download complete. Extracting...')
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Extracted completely to {extract_dir}.')
