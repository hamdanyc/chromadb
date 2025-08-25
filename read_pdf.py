import PyPDF2
import os
from chromadb import CloudClient

def process_pdf(pdf_path):
    # 1. Read PDF and extract text
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text_pages = [page.extract_text() for page in reader.pages]
    
    # 2. Create documents with metadata
    documents = []
    for i, text in enumerate(text_pages):
        metadata = {
            "page_number": i + 1,
            "word_count": len(text.split())
        }
        documents.append({
            "text": text,
            "metadata": metadata
        })
    
    # 3. Create collection using PDF filename
    collection_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 4. Add to Chroma Cloud using CloudClient
    client = CloudClient(
        tenant=os.getenv("CH_TENANT"),
        database=os.getenv("CH_DATABASE"),
        api_key=os.getenv("CH_API_KEY")
    )
    
    collection = client.create_collection(name=collection_name)
    
    # 5. Add documents to collection
    for doc in documents:
        collection.add(
            documents=doc["text"],
            metadatas=doc["metadata"],
            ids=f"{collection_name}_page_{doc['metadata']['page_number']}"
        )
    
    return collection_name
