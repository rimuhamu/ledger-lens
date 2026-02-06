import os
import sys

sys.path.append("src")

import requests
from database import Database
from dotenv import load_dotenv

load_dotenv()

def setup_project_data():
    # Define paths
    raw_data_dir = "data/raw"
    db_dir = "data/vectorstore"
    
    # Create directories if they don't exist
    for folder in [raw_data_dir, db_dir]:
        os.makedirs(folder, exist_ok=True)
        print(f"Verified directory: {folder}")

    # Download the BCA 2024 Report
    url = "https://www.bca.co.id/-/media/Feature/Report/File/S8/Laporan-Tahunan/2025/20250306-BCA-AR-2024-ENG.pdf"
    target_path = os.path.join(raw_data_dir, "BCA_AR_2024.pdf")

    if not os.path.exists(target_path):
        print("Downloading BCA 2024 Annual Report...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully saved to {target_path}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    else:
        print("BCA 2024 Report already exists. Skipping download.")

    print("Initializing Vector Database (ChromaDB)...")
    db = Database(db_path=db_dir)

    if not os.listdir(db_dir):
        success = db.ingest_document(target_path)
        if success:
            print("Setup Complete: Data downloaded and indexed.")
        else:
            print("Setup Error: Ingestion failed.")
    else:
        print("Vector store already exists. Setup skipped.")

if __name__ == "__main__":
    setup_project_data()