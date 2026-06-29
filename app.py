import streamlit as st
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import load_embedding_model
from utils.vector_store import create_vector_store
from utils.rag_chain import (
    load_vector_store,
    retrive_documents,
    generate_response
)

#page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

#session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pages" not in st.session_state:
    st.session_state.pages = None
if "text" not in st.session_state:
    st.session_state.text = None

#Title
st.title("PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")

#Upload PDF
uploaded_file = st.file_uploader(
    "Choose your PDF file",
    type="pdf"
)

#Reset Session when PDF is removed
if uploaded_file is None:
    st.session_state.pdf_processed = False
    st.session_state.embedding_model = None
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.pages = None
    st.session_state.text = None
    st.session_state.documents = None
    st.session_state.embeddings = None

#Process Button
if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    if st.button("process PDF"):
        with st.spinner("Reading PDF..."):
            documents, pages = load_pdf(uploaded_file)
            chunks = split_text(documents)
            embedding_model = load_embedding_model()
            embeddings = embedding_model.embed_documents(
                [doc.page_content for doc in chunks]
            )
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            #save everything in Session state
            st.session_state.documents = documents
            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.embedding_model = embedding_model
            st.session_state.vector_store = vector_store
            st.session_state.pdf_processed = True
            vector_store.save_local("vectorstore")
        st.success("PDF processed successfully!")

#Show PDF Information
if  uploaded_file is not None and st.session_state.pdf_processed:
        
        #st.subheader("File Details")
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"Total Pages: {st.session_state.pages}")
        #st.subheader("Extracted Text Preview")
        #st.write(st.session_state.text[:1000])
        #st.subheader("Chunk Information")
        st.write(f"Total Chunks: {len(st.session_state.chunks)}")
        #st.write("**First Chunk**")
        #st.write(st.session_state.chunks[0])
        #st.subheader("Embedding Information")
        #st.write(f"Total Embeddings: {len(st.session_state.embeddings)}")
        #st.write(f"Embedding Dimension: {len(st.session_state.embeddings[0])}")
        #st.write("### First 10 Values")
        #st.write(st.session_state.embeddings[0][:10])
        #st.success("Vector Database Created Successfully!")

#Ask Questions
if uploaded_file is not None and st.session_state.pdf_processed:
    st.write("---")
    st.header("Ask Questions")
    question = st.text_input(
        "Enter your question"
    )
    if question:
        retrieved_docs = retrive_documents(
            st.session_state.vector_store,
            question
        )
        # with st.expander("📚 View Retrieved Sources"):
        #     pages = sorted(
        #         set(
        #             doc.metadata["page"] + 1
        #             for doc in retrieved_docs
        #         )
        #     )
        #     st.markdown("### Source Pages")
        #     for page in pages:
        #         st.write(f"Page {page}")
        #     st.divider()
        #     for i, doc in enumerate(retrieved_docs):
        #         page = doc.metadata["page"] + 1
        #         st.markdown(f"### Chunk {i+1} (page {page})")
        #         st.write(doc.page_content)
        #         st.divider()
        with st.spinner("Generating Answer..."):
            answer = generate_response(
                question,
                retrieved_docs
            )
        st.subheader("🤖 AI Response")
        st.info(answer)

        pages = sorted(
            set(
                doc.metadata["page"] + 1
                for doc in retrieved_docs
            )
        )

        if len(pages) == 1:
            st.caption(f"📄 Source: Page {pages[0]}")
        else:
            st.caption(f"📄 Sources: Pages {', '.join(map(str, pages))}")

        with st.expander("📚 View Retrieved Sources"):
            # pages = sorted(
            #     set(
            #         doc.metadata["page"] + 1
            #         for doc in retrieved_docs
            #     )
            # )
            st.markdown("### 📄 Source Pages")
            for page in pages:
                st.write(f" 📄 Page {page}")
            st.divider()
            for i, doc in enumerate(retrieved_docs):
                page = doc.metadata["page"] + 1
                st.markdown(f"### Chunk {i+1} (page {page})")
                st.write(doc.page_content)
                st.divider()