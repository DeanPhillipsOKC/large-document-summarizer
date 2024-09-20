from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import streamlit as st

class FlowingSummaryNarrativeCreator:
    def __init__(self):
        self.llm = self._create_llm()
        self.output_parser = self._create_output_parser()
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()

    def _create_llm(self):
        return ChatOpenAI(temperature=0, model="gpt-4o")
    
    def _create_output_parser(self):
        return StrOutputParser()
    
    def _create_prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are a highly skilled assistant that excels in summarizing text. Your goal is to create a coherent and detailed summary of a given passage,
                    focusing on the main events, characters, and themes. Your summary should flow seamlessly, as if the passage naturally progresses into the summary.
                    Avoid using introductory phrases like 'In this passage' and directly address the narrative or descriptions from the text.
                    Your result must be detailed and at least 2 paragraphs long.  Avoid using first person"""
                ),
                HumanMessagePromptTemplate.from_template("Passage:\n{text}")
            ]
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc.page_content)

    st.cache_data()
    def create_summary(self, docs):
        total_tokens = 0
        summary = ""

        for doc in docs:
            # Get the new summary.
            total_tokens = total_tokens + self._get_num_tokens(doc)
            new_summary = self.chain.invoke({"text": doc.page_content})
            # Update the list of the last two summaries: remove the first one and add the new one at the end.
            summary+=new_summary

        return {
            "summary": summary,
            "total_tokens_sent": total_tokens
        }
