from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class FlowingOutlineCreator:
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
                SystemMessagePromptTemplate.from_template(
                    """You are a highly skilled assistant that excels in creating outlines for books.
                    Your outlines capture the plot elements, characters, thematic elements, major dialog, and major thoughts of the characters (or the author). Your 
                    goal is to capture the important parts of the book without the bloat so that it can be consumed efficiently without sacrificing too much knowledge
                    or wisdom that the text imparts.  It should work for novels, short stories, and non-fiction books alike.
                    Avoid using introductory phrases like 'In this passage' and directly address the narrative or descriptions from the text."""
                ),
                HumanMessagePromptTemplate.from_template("Passage:\n{text}\n\nFormat instrictions: Use nested bulleted lists in markdown format.")
            ]
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc.page_content)

    def create_outline(self, docs):
        total_tokens = 0
        outline = ""

        for doc in docs:
            total_tokens = total_tokens + self._get_num_tokens(doc)
            new_outline = self.chain.invoke({"text": doc.page_content})
            outline+=new_outline

        return {
            "outline": outline,
            "total_tokens_sent": total_tokens
        }
