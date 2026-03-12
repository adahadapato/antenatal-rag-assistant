# Initialize.py
import os

folder_path = r"C:\Projects\Hakathon"
os.makedirs(folder_path, exist_ok=True)

doc_path = os.path.join(folder_path, "ANTENATAL CARE SCHEDULE.pdf")
vector_db_path = os.path.join(folder_path, "hakathon_2026_chroma_store")
augmented_db_path = os.path.join(folder_path, "argumentation.db")
output_file = os.path.join(folder_path, "extracted_pdf_text.txt")
augmented_db_json = os.path.join(folder_path, "augment_db3.json")

if __name__ == "__main__":
    print("Folder ready:", folder_path)
    print("PDF path:", doc_path)
    print("Vector DB path:", vector_db_path)
    print("Augmentted DB path:", augmented_db_path)