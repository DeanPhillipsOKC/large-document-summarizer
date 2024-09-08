import streamlit as st
from docs import WebDocumentRetriever
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd
import numpy as np
import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import FlowingSummaryNarrativeCreator

class LargeDocumentSummarizerApp:
    def main(self):
        st.title("Large Document Summarizer")
        url = st.text_input("Enter the URL of the document")
        summarize_button = st.button("Summarize")

        if summarize_button:
            model = ChatOpenAI(temperature=0)

            doc_contents = WebDocumentRetriever.retrieve(url)
            num_tokens_in_original = model.get_num_tokens(doc_contents)

            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

            text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="interquartile")
            docs = text_splitter.create_documents([doc_contents])

            embeddings = embedding_function.embed_documents([doc.page_content for doc in docs])

            content_list = [doc.page_content for doc in docs]
            df = pd.DataFrame(content_list, columns=['page_content'])

            vectors = [embedding for embedding in embeddings]
            array = np.array(vectors)
            embeddings_series = pd.Series(list(array))
            df['embeddings'] = embeddings_series
            
            array = array.astype('float32') 
            num_chunks = len(docs)

            # num_clusters = min(50, num_chunks)
            num_clusters = num_chunks / 2

            dimension = array.shape[1]
            kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=True)
            kmeans.train(array)
            centroids = kmeans.centroids
            index = faiss.IndexFlatL2(dimension)
            index.add(array)

            D, I = index.search(centroids, 1)

            sorted_array = np.sort(I, axis=0)
            sorted_array=sorted_array.flatten()
            extracted_docs = [docs[i] for i in sorted_array]

            flowing_summary_narrative_creator = FlowingSummaryNarrativeCreator()
            result = flowing_summary_narrative_creator.create_summary(extracted_docs)
            summary = result["summary"]
            total_tokens = result["total_tokens_sent"]

            premium_model = ChatOpenAI(temperature=0, model="gpt-4o")
            finalizer_prompt = ChatPromptTemplate.from_template(""" 
                You are a professional editor and writer. You will be given a summary of a book. Your task is to refine the summary, make it more detailed,
                compelling, and less redundant.
                                                                
                Passage:
                                                                
                {text}
                                                                
                SUMMARY:
            """)

            final_chain = finalizer_prompt | premium_model | StrOutputParser()

            final_summary = final_chain.invoke({"text": summary})

            stats = [
                ['Tokens in original document', num_tokens_in_original],
                ['Tokens sent to OpenAI', total_tokens],
                ['Tokens in rough summary', model.get_num_tokens(summary)],
                ['Tokens in final summary', model.get_num_tokens(final_summary)]
            ]
            stats_df = pd.DataFrame(stats, columns=['Stat', 'Value'])
            st.table(stats_df)

            st.subheader("Summary")
            st.write(final_summary)

if __name__ == "__main__":
    app = LargeDocumentSummarizerApp()
    app.main()