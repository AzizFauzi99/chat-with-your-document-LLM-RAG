import os
import openai
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

sys.path.append('../..')

import panel as pn  
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

openai.api_key = os.environ['OPENAI_API_KEY']


pdf_file_path = "asset/All_About_JKT48.pdf"
# options=["stuff", "map_reduce", "refine", "map_rerank"]
chain_type = "refine"
k = 3 

# --- load_db function ---
def load_db():
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during PDF loading: {e}")
        return None



# --- Chatbot Interface ---
class ChatbotInterface:
    def __init__(self):
        self.qa_chain = load_db()
        self.chat_interface = pn.chat.ChatInterface(callback=self.get_response, max_height=500)
        if self.qa_chain:
            self.chat_interface.send("Ask me anything!", user="Assistant", respond=False)
        else:
            self.chat_interface.send("Error loading PDF. Please check the file path and your OpenAI API Key.", user="Assistant", respond=False)

    def get_response(self, contents, user, instance):
        if self.qa_chain:
            result = self.qa_chain({"question": contents, "chat_history": []})
            response = result["answer"]
        else:
            response = "PDF loading failed."

        for index in range(len(response)):
            yield response[0:index + 1]



# --- Panel Layout and Widgets ---
chatbot_interface = ChatbotInterface()

main_layout = pn.Column(
    chatbot_interface.chat_interface,
    width=700,
)

main_layout.servable()