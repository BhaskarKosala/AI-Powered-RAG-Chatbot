from langchain_huggingface import HuggingFaceEmbeddings
def load_embedding_model():
    '''Loading BGE Embedding model'''
    embedding_model = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )
    return embedding_model