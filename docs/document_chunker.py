from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import streamlit as st

class DocumentChunker:
    @staticmethod
    @st.cache_data()
    def chunk_document(doc):
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

        text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="interquartile")
        docs = text_splitter.create_documents([doc])

        return docs