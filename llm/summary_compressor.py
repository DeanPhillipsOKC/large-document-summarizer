from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class SummaryCompressor:
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
                You are a professional editor and writer. You will be given a summary of a book. Your task is to compress it by removing extraneous words,
                using smaller synonyms, and removing any redundancies. You should not remove important facts about the characters, themes, plot, or ideas."""),
                HumanMessagePromptTemplate.from_template("Passage:\n\n{text}")
            ]
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc)

    def create_summary(self, doc):
        summary = self.chain.invoke({"text": doc})
        tokens_sent = self._get_num_tokens(doc)
        return {
            "summary": summary,
            "total_tokens_sent": tokens_sent
        }

