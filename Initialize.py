# Initialize.py
import os

folder_path = r"C:\Projects\Hakathon"
os.makedirs(folder_path, exist_ok=True)

doc_path = os.path.join(folder_path, "ANTENATAL CARE SCHEDULE.pdf")
db_path = os.path.join(folder_path, "hakathon_2026_chroma_store")

if __name__ == "__main__":
    print("Folder ready:", folder_path)
    print("PDF path:", doc_path)
    print("DB path:", db_path)