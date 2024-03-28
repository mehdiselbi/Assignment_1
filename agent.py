import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import openai

class Agent:
    def __init__(self, openai_api_key:str | None = None) -> None:
        """
        Initialize the Agent with an OpenAI API key.
        
        Parameters:
        openai_api_key (str | None): The OpenAI API key used for authentication with OpenAI services.
        
        Outputs:
        None
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = openai.OpenAI(temperature=0,openai_api_key=openai_api_key)
        self.chat_history = None
        self.chain = None
        self.db = None

    def ask(self,question:str) -> str:
        """
        Process a user's question and return the agent's response.
        
        Parameters:
        question (str): The user's question or query.
        
        Returns:
        str: The agent's response to the question.
        """
        if self.chain is None:
            response = "Please add a PDF Document."
        else:
            response = self.chain({"question": question, "chat_history":self.chat_history})
            response = response["answer"].strip()
            self.chat_history.append((question, response))
        return response
    
    def doc_load(self, path:str) -> None:
        """
        Load a document from a specified path for processing.
        
        Parameters:
        path (str): The filesystem path to the document to be loaded.
        
        Outputs:
        None
        """
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        splitted_documents = self.textsplitter.split_documents(documents)

        if self.db is None:
            self.db = faiss.FAISS.from_documents(splitted_documents, self.embeddings)
            self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())
            self.chat_history = []
        else:
            self.db.add_documents(splitted_documents)
    
    def forget(self) -> None:
        """
        Reset the agent's knowledge base and conversation history.
        
        This method clears the document database, the conversational chain, and the chat history,
        effectively resetting the agent's state.
        """
        self.db = None
        self.chain = None
        self.chat_history = None