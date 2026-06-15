import os
import pickle
import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreateBM25")

PDF_DIR = "PDFs"
BM25_PATH = "bm25_index.pkl"
TEXTS_PATH = "all_texts.pkl"

def main():
    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory {PDF_DIR} not found.")
        return

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdfs)} PDF files.")

    documents = []
    for pdf in pdfs:
        filepath = os.path.join(PDF_DIR, pdf)
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            text = text.strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": pdf}))
        except Exception as e:
            logger.error(f"Error reading {pdf}: {e}")

    if not documents:
        logger.error("No text extracted from documents.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    all_texts = [doc.page_content for doc in chunks]
    all_metadatas = [doc.metadata for doc in chunks]

    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump((all_texts, all_metadatas), f)

    logger.info("Local BM25 index saved successfully.")

if __name__ == "__main__":
    main()
