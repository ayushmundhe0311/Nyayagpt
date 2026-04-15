import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
PDF_FOLDER_PATH = "./data"
PERSIST_DIRECTORY = "./local_chroma_db"

def run_ingestion():
    print(" Starting PDF Ingestion...")
    
    # Create data folder if it doesn't exist
    if not os.path.exists(PDF_FOLDER_PATH):
        os.makedirs(PDF_FOLDER_PATH)
        print(f"Created {PDF_FOLDER_PATH} folder. Please add PDFs there and run again.")
        return

    # 1. Load PDFs
    all_documents = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(" No PDF files found in the ./data folder.")
        return

    for file_name in pdf_files:
        file_path = os.path.join(PDF_FOLDER_PATH, file_name)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f" Loaded: {file_name} ({len(documents)} pages)")

    # 2. Split Text (Optimized for Indian Legal Documents)
    legal_separators = [
        r"\nSection\s+\d+", 
        r"\nArticle\s+\d+", 
        r"\nCHAPTER\s+[IVXLCDM]+",
        "\n\n", 
        "\n"
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        is_separator_regex=True,
        separators=legal_separators
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f" Split into {len(chunks)} text chunks.")

    # 3. Create & Persist Vector DB
    print("Generating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f" Vector database successfully saved to {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    run_ingestion()