from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents):
    '''splits documents into chunks'''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = splitter.split_documents(documents)
    return chunks