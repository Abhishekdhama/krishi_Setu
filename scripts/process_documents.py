import os
import fitz  # This is the name for the PyMuPDF library

def extract_text_from_pdfs(documents_folder, output_folder):
    print("--- Starting PDF Text Extraction Process ---")

    if not os.path.exists(documents_folder):
        print(f"Error: The '{documents_folder}' does not exist. Please create it and add your PDF files.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{documents_folder}'.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(documents_folder, pdf_file)
            print(f"  -> Processing: {pdf_file}")

            doc = fitz.open(file_path)
            
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            
            doc.close()

            base_filename = os.path.splitext(pdf_file)[0]
            output_txt_filename = f"{base_filename}.txt"
            output_txt_path = os.path.join(output_folder, output_txt_filename)

            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            print(f"      Successfully extracted text to '{output_txt_filename}'")

        except Exception as e:
            print(f"      Error processing {pdf_file}: {e}")

    print("\n--- PDF Text Extraction Process Finished ---")

if __name__ == "__main__":
    docs_input_dir = 'documents'
    text_output_dir = 'knowledge_base_text'
    extract_text_from_pdfs(docs_input_dir, text_output_dir)