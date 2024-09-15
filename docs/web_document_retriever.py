import requests
from bs4 import BeautifulSoup
import streamlit as st

class WebDocumentRetriever:
    @staticmethod
    @st.cache_data()
    def retrieve(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        contents = "".join(paragraph.get_text() for paragraph in paragraphs)
        return contents