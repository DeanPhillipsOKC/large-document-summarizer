from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class FinalSummaryNarrativeCreator:
    def __init__(self):
        self.llm = self._create_llm()
        self.output_parser = self._create_output_parser()
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()

    def _create_llm(self):
        return ChatOpenAI()
    
    def _create_output_parser(self):
        return StrOutputParser()
    
    def _create_prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(""" 
                You are a professional editor and writer. You will be given a summary of a book. Your task is to refine the summary, make it more detailed,
                compelling, and less redundant.  Do not worry about spoilers.  All important plot and thematic elements need to be captured.  Think of it
                as a more efficient way of consuming the text than an actual summary."""),
                HumanMessagePromptTemplate.from_template("Passage:\n\n{text}Format Instructions: Use markdown format")
            ]
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc.page_content)

    def create_summary(self, doc):
        return self.chain.invoke({"text": doc})
