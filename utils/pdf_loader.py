from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(uploaded_file):
    '''It reads the uploaded pdf and returns:
    1. Langchain Documents (one document per page)
    2. Total Pages
    '''
    reader = PdfReader(uploaded_file)
    documents = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "source": uploaded_file.name
                    }
                )
            )
    return documents, len(reader.pages)