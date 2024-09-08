from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate


class FlowingSummaryNarrativeCreator:
    def __init__(self):
        self.llm = self._create_llm()
        self.output_parser = self._create_output_parser()
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()

    def _create_llm(self):
        return ChatOpenAI(temperature=0)
    
    def _create_output_parser(self):
        return StrOutputParser()
    
    def _create_prompt_template(self):
        return ChatPromptTemplate.from_template("""
            You will be given different passages from a book one by one. Provide a summary of the following text. Your result must be detailed and 
            atleast 2 paragraphs. When summarizing, directly dive into the narrative or descriptions from the text without using introductory 
            phrases like 'In this passage'. Directly address the main events, characters, and themes, encapsulating the essence and significant 
            details from the text in a flowing narrative. The goal is to present a unified view of the content, continuing the story seamlessly as 
            if the passage naturally progresses into the summary

            Passage:

            ```{text}```
            SUMMARY:
            """
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc.page_content)

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
