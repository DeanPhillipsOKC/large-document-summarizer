import streamlit as st
from docs import WebDocumentRetriever
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd
from llm import FlowingSummaryNarrativeCreator
from llm import FinalSummaryNarrativeCreator
from vector import DocumentSimilarityClusterer

class LargeDocumentSummarizerApp:
    def main(self):
        st.title("Large Document Summarizer")
        url = st.text_input("Enter the URL of the document")
        cluster_multiplier = st.slider(
            "Cluster multiplier", min_value= 0.1, max_value=1.0, value=0.5, step=0.05, 
            help="The number of clusters used will be equal to the number of chunks in the semantically parsed document, times this value."
        )
        st.latex(f"k=\lfloor {cluster_multiplier}n \\rfloor")
        summarize_button = st.button("Summarize")

        if summarize_button:
            model = ChatOpenAI(temperature=0)

            doc_contents = WebDocumentRetriever.retrieve(url)
            num_tokens_in_original = model.get_num_tokens(doc_contents)

            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

            text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="interquartile")
            docs = text_splitter.create_documents([doc_contents])

            embeddings = embedding_function.embed_documents([doc.page_content for doc in docs])

            clustered_docs = DocumentSimilarityClusterer.cluster_documents(docs, embeddings, cluster_multiplier)

            flowing_summary_narrative_creator = FlowingSummaryNarrativeCreator()
            result = flowing_summary_narrative_creator.create_summary(clustered_docs)
            summary = result["summary"]
            total_tokens = result["total_tokens_sent"]

            final_summary_narrative_creator = FinalSummaryNarrativeCreator()
            final_summary = final_summary_narrative_creator.create_summary({"text": summary})

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