import streamlit as st
from docs import WebDocumentRetriever, DocumentChunker
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
        file = st.file_uploader("upload file")
        html = None

        if file:
            html = file.getvalue()
        load_button = st.button("Load document", use_container_width=True)

        if load_button:
            st.session_state["document_loaded"] = True

        if "document_loaded" in st.session_state.keys() and st.session_state["document_loaded"]:
            model = ChatOpenAI()

            doc_contents = WebDocumentRetriever.retrieve(url, html)
            num_tokens_in_original = model.get_num_tokens(doc_contents)

            docs = DocumentChunker.chunk_document(doc_contents)

            st.write(f"Number of chunks (n): {len(docs)}")

            use_clustering = st.checkbox("Use clustering", value=True)

            if use_clustering:
                cluster_multiplier = st.slider(
                    "Cluster multiplier (k)", min_value= 0.01, max_value=1.0, value=0.5, step=0.01, 
                    help="The number of clusters used will be equal to the number of chunks in the semantically parsed document, times this value."
                )
                st.latex(f"k=\lfloor {cluster_multiplier}n \\rfloor = {int(len(docs) * cluster_multiplier)}")
            
            llm_processing_method = st.radio(
                "LLM processing method",
                ["Summary"],
                captions=["Summarize each cluster and then summarize the summaries.", "Summarize each cluster and then outline the summaries."]
            )

            summarize_button = st.button("Summarize", use_container_width=True)

            if summarize_button:

                progress_bar = st.progress(0)

                if use_clustering:
                    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
                    progress_bar.progress(30, text="Converting chunks into vectors.")
                    embeddings = embedding_function.embed_documents([doc.page_content for doc in docs])

                    progress_bar.progress(50, text="Clustering the document vectors using a k-means algorithm.")
                    clustered_docs = DocumentSimilarityClusterer.cluster_documents(docs, embeddings, cluster_multiplier)
                else:
                    clustered_docs = docs

                summary = ""
                final_summary = ""

                if llm_processing_method == "Summary":
                    flowing_summary_narrative_creator = FlowingSummaryNarrativeCreator()
                    progress_bar.progress(60, text="Creating a flowing summary of the chunks (this might take awhile)")
                    result = flowing_summary_narrative_creator.create_summary(clustered_docs)
                    summary = result["summary"]
                    summary = summary.replace('.#', '.\n#')
                    print("Original summary:\n\n")
                    print(summary)
                    total_tokens = result["total_tokens_sent"]

                    with st.expander("Original summary"):
                        st.markdown(summary)

                    final_summary_narrative_creator = FinalSummaryNarrativeCreator()
                    progress_bar.progress(80, text="Creating a final summary of the document")
                    final_summary = final_summary_narrative_creator.create_summary(summary)

                progress_bar.progress(100, text="Adding statistics.")
                stats = [
                    ['Tokens in original document', num_tokens_in_original],
                    ['Tokens sent to OpenAI (original)', total_tokens],
                    ['Tokens in rough summary', model.get_num_tokens(summary)],
                    ['Tokens in final summary', model.get_num_tokens(final_summary)]
                ]
                stats_df = pd.DataFrame(stats, columns=['Stat', 'Value'])
                st.table(stats_df)

                with st.expander("Compressed summary"):
                    st.markdown(final_summary)

if __name__ == "__main__":
    app = LargeDocumentSummarizerApp()
    app.main()