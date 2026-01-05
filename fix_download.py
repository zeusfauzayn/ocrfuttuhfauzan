import os
import shutil
import requests
import sys
from tqdm import tqdm

def fix_emnist():
    # Helper to find where emnist package stores data
    user_home = os.path.expanduser('~')
    cache_dir = os.path.join(user_home, '.cache', 'emnist')
    
    print(f"Checking cache directory: {cache_dir}")
    
    # Clean up existing corrupted files
    if os.path.exists(cache_dir):
        print("Removing corrupted cache...")
        shutil.rmtree(cache_dir)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    filename = "gzip.zip"
    filepath = os.path.join(cache_dir, filename)
    
    print(f"Downloading EMNIST from {url}...")
    print("This is a large file (~536 MB), please be patient...")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
        print("\nDownload complete!")
        print(f"File saved to: {filepath}")
        
    except Exception as e:
        print(f"\nError downloading: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            os.remove(filepath)
            
if __name__ == "__main__":
    fix_emnist()
