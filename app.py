import streamlit as st
from docs import WebDocumentRetriever
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd
from llm import FlowingSummaryNarrativeCreator, FinalSummaryNarrativeCreator, FlowingOutlineCreator, FinalOutlineCreator, SummaryCompressor
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
        
        llm_processing_method = st.radio(
            "LLM processing method",
            ["Summary", "Outline"],
            captions=["Summarize each cluster and then summarize the summaries.", "Summarize each cluster and then outline the summaries."]
        )

        summarize_button = st.button("Summarize")

        if summarize_button:
            model = ChatOpenAI()

            progress_bar = st.progress(0)

            progress_bar.progress(10, text="Retrieving the document")
            doc_contents = WebDocumentRetriever.retrieve(url)
            num_tokens_in_original = model.get_num_tokens(doc_contents)

            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

            progress_bar.progress(20, text="Semantically chunking the document")
            text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="interquartile")
            docs = text_splitter.create_documents([doc_contents])

            progress_bar.progress(30, text="Converting chunks into vectors.")
            embeddings = embedding_function.embed_documents([doc.page_content for doc in docs])

            progress_bar.progress(50, text="Clustering the document vectors using a k-means algorithm.")
            clustered_docs = DocumentSimilarityClusterer.cluster_documents(docs, embeddings, cluster_multiplier)

            summary = ""
            final_summary = ""

            if llm_processing_method == "Summary":
                progress_bar.progress(60, text="Creating a flowing summary of the chunks (this might take awhile)")
                flowing_summary_narrative_creator = FlowingSummaryNarrativeCreator()
                result = flowing_summary_narrative_creator.create_summary(clustered_docs)
                final_summary = result["summary"]
                final_summary = final_summary.replace('.#', '.\n#')
                total_tokens = result["total_tokens_sent"]

                # progress_bar.progress(80, text="Creating a final summary of the document")
                # final_summary_narrative_creator = FinalSummaryNarrativeCreator()
                # final_summary = final_summary_narrative_creator.create_summary(summary)
            else:
                progress_bar.progress(60, text="Creating a flowing outline of the chunks")
                flowing_outline_creator = FlowingOutlineCreator()
                result = flowing_outline_creator.create_outline(clustered_docs)
                summary = result["outline"]

                progress_bar.progress(80, text="Creating a final outline of the document")
                total_tokens = result["total_tokens_sent"]
                final_outline_creator = FinalOutlineCreator()
                final_summary = final_outline_creator.create_outline({"text": summary})

            progress_bar.progress(100, text="Adding statistics.")
            stats = [
                ['Tokens in original document', num_tokens_in_original],
                ['Tokens sent to OpenAI (original)', total_tokens],
                ['Tokens in rough summary', model.get_num_tokens(summary)],
                ['Tokens in final summary', model.get_num_tokens(final_summary)]
            ]
            stats_df = pd.DataFrame(stats, columns=['Stat', 'Value'])
            st.table(stats_df)

            st.markdown(final_summary)

if __name__ == "__main__":
    app = LargeDocumentSummarizerApp()
    app.main()