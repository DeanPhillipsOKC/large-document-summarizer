from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class FinalOutlineCreator:
    def __init__(self):
        self.llm = self._create_llm()
        self.output_parser = self._create_output_parser()
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()

    def _create_llm(self):
        return ChatOpenAI(model="gpt-4o")
    
    def _create_output_parser(self):
        return StrOutputParser()
    
    def _create_prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(""" 
                You are a professional editor and writer. You will be given an outline of a book.  Your goal is to revise the outline so that it flows
                coherently.  You should also avoid redundancies as much as possible.  It should capture the important parts of the book outline without
                too much bloat.  The reader wants to use your outline to consume the wisdom of the text efficiently, not necessarily for enjoyment."""),
                HumanMessagePromptTemplate.from_template("Passage:\n{text}\n\nFormat instructions: A nested bulleted list in markdown format.")
            ]
        )
    
    def _create_chain(self):
        return self.prompt_template | self.llm | self.output_parser

    def _get_num_tokens(self, doc):
        return self.llm.get_num_tokens(doc.page_content)

    def create_outline(self, doc):
        return self.chain.invoke({"text": doc})
