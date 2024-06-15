import os
import getpass
import re
import datetime
import requests
import uuid
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Union

# Type of LangChain chains
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain

# LangChain with ConversationBufferMemory and PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Vector Store with Facebook AI Similarity Search (FAISS) based on L2 distance
# from langchain.vectorstores import FAISS will be deprecated
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# LangChain processing and runnables
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create Retrieval and Documents chains
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# NVIDIA NeMo GuardRails
import nest_asyncio
# Running inside a notebook, patch the AsyncIO loop
nest_asyncio.apply()

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Documents that are to be embedded and stored in Vector Database
# List of web pages containing NVIDIA Triton technical documentation
raw_documents = [
    {
        "title": "NVIDIA Triton Inference Server",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html"},
    {
        "title": "NVIDIA Triton Quickstart",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html"},
    {
        "title": "NVIDIA Triton Model Repository",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html"},
    {
        "title": "NVIDIA Triton Model Analyzer",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html"},
    {
        "title": "NVIDIA Triton Model Analyzer",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html"}
]

# Relative embedding path for vectorstore
embedding_path = "./embed"

# Configuration path for NVIDIA NeMo Guardrails
guardrails_config_path = "./config"

class Vectorstore:
    """
    A class representing a collection of documents indexed into a vectorstore.

    Parameters:
    raw_documents (list): A list of dictionaries representing the sources of the raw documents. Each dictionary should have 'title' and 'url' keys.

    Attributes:
    raw_documents (list): A list of dictionaries representing the raw documents.
    documents (list): A list of documents representing extracted texts from HTML documents.
    texts (int): A list of the associated document chunks for the embeddings.
    docsearch (list): Embeddings from texts using Facebook AI Similarity Search (FAISS).

    Methods:
    html_document_loader(): Loads the data from the data sources and extract texts from the HTML content.
    create_embeddings(): Embeds the document chunks using the NVIDIA AI Endpoints and save embeddings to offline vector store in the
                         ./embed directory for future re-use.
    index_docs(): Generate embeddings with "NV-Embed-QA" model and save locally with FAISS library for efficient similarity search
                  and clustering of dense vectors.
    """

    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents

        # Create embeddings
        self.create_embeddings(embedding_path=embedding_path)

    def html_document_loader(self, url: Union[str, bytes]) -> str:
        """
        Loads the HTML content of a document from a given URL and return it's content.
    
        Args:
            url: The URL of the document.
    
        Returns:
            The content of the document.
    
        Raises:
            Exception: If there is an error while making the HTTP request.
    
        """
        try:
            response = requests.get(url)
            html_content = response.text
        except Exception as e:
            print(f"Failed to load {url} due to exception {e}")
            return ""
    
        try:
            # Create a Beautiful Soup object to parse html
            soup = BeautifulSoup(html_content, "html.parser")
    
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
    
            # Get the plain text from the HTML document
            text = soup.get_text()
    
            # Remove excess whitespace and newlines
            text = re.sub("\s+", " ", text).strip()
    
            return text
        except Exception as e:
            print(f"Exception {e} while loading document")
            return ""
    
    def index_docs(self, url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir) -> None:
        """
        Split the document into chunks and create embeddings for the document
    
        Args:
            url: Source url for the document.
            splitter: Splitter used to split the document
            documents: list of documents whose embeddings needs to be created
            dest_embed_dir: destination directory for embeddings
    
        Returns:
            None
        """
        embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")  # nvolveqa_40k
        
        for document in documents:
            texts = splitter.split_text(document.page_content)
    
            # metadata to attach to document
            metadatas = [document.metadata]
    
            # create embeddings and add to vector store
            if os.path.exists(dest_embed_dir):
                # print("2. Update embeddings...")
                update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings,
                                          allow_dangerous_deserialization=True)
                update.add_texts(texts, metadatas=metadatas)
                update.save_local(folder_path=dest_embed_dir)
            else:
                print("2. Create and save embeddings...")
                # Save embeddings to offline vector store
                docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                docsearch.save_local(folder_path=dest_embed_dir)

    def create_embeddings(self, embedding_path: str):
    
        print(f"Storing embeddings to {embedding_path}")
    
        documents = []
    
        print("1. Loading documents...")
        for raw_document in self.raw_documents:  # urls
            raw_document_url = raw_document["url"]
            document = self.html_document_loader(url=raw_document_url)
            documents.append(document)
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )

        # Chunking documents
        texts = text_splitter.create_documents(documents)
        self.index_docs(raw_document_url, text_splitter, texts, embedding_path)
        print("Generated embedding successfully")

# Single prompt at a time for streamlit LLM app
class Chatbot:
    def __init__(self, model='meta/llama3-8b-instruct'):
        """
        Initializes an instance of the Chatbot class.

        """

        # GPU-accelerated generation of text embeddings used for question-answering retrieval (embed-qa-4)
        # Create the embeddings model using NVIDIA Retrieval QA Embedding endpoint
        self.embedding_model = NVIDIAEmbeddings(model="NV-Embed-QA")

        # Load documents from vector database using FAISS (Facebook AI Similarity Search) quickly search for embeddings of
        # multimedia documents that are similar to each other.
        # Embed documents in directly with embedding_model
        print("Load the embeddings with FAISS and setting the API access for LLMs ...\n")
        self.docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=self.embedding_model,
                                          allow_dangerous_deserialization=True)
        self.retriever = self.docsearch.as_retriever()
        self.conversation_id = "Begin..."

        # Get the required API keys from the environment variables
        if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
            nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
            assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
            os.environ["NVIDIA_API_KEY"] = nvapi_key

        # Use for Anthrophic Claude foundation models
        if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
            antapi_key = getpass.getpass("Enter your ANTHROPIC API key: ")
            os.environ["ANTHROPIC_API_KEY"] = antapi_key

        # Use for guardrails in config.yml
        if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-proj-"):
            openaiapi_key = getpass.getpass("Enter your OPENAI API key: ")
            os.environ["OPENAI_API_KEY"] = openaiapi_key

        # Create ChatNVIDIA based on selected Foundational Model
        print(f"Create ChatNVIDIA based on selected Foundational Model : {model}")
        self.selected_fm = ChatNVIDIA(
                                      model=model,  # "google/gemma-7b",
                                      temperature=0.1,
                                      top_p=0.7,
                                      top_k=250,
                                      max_tokens=1000,  # 1024

                                      system="You are BC, an AI assistant model 2701 created by SAM. \
                                              You are an expert in Large Language Model, Generative AI and Graphics Processing Unit (GPU). \
                                              You should say you do not know if you do not know and answer only if you are very confident. \
                                              Answer in number bulleted format.",
                                      )

        # Stores the chat history
        self.chat_history = []

        # Save the RAG chat history
        self.prev_docs_chat_history_response = []

    def create_rag_prompt_templates(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the \
                                           chat history, formulate a standalone question which can be understood without the chat history. \
                                           Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                                    [
                                        ("system", contextualize_q_system_prompt),
                                        MessagesPlaceholder("chat_history"),
                                        ("human", "{input}"),
                                    ]
                                 )

        qa_system_prompt = """You are BC, an AI assistant model 2701 created by SAM. \
                              You are an expert in Large Language Model, Generative AI and Graphics Processing Unit (GPU). \
                              You should say you do not know if you do not know and answer only if you are very confident. \
                              Answer in number bulleted format.\

                              {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
                                    [
                                        ("system", qa_system_prompt),
                                        MessagesPlaceholder("chat_history"),
                                        ("human", "{input}"),
                                    ]
                    )
         
        return contextualize_q_prompt, qa_prompt

    # Implement for Guardrails to overcome "TypeError: Object of type HumanMessage is not JSON serializable" with rag_chain
    def role_message(self, role: str, message: str):
        return {"role": role, "content": message}

    def add_chat_history(self, chat_history, question, rag_guardrails_response):
        dict_role_message = self.role_message("human", question)
        chat_history.append(dict_role_message)

        if "answer" not in rag_guardrails_response.keys():
            # Activate guardrails
            question_response = rag_guardrails_response["output"]
        else:
            # With "output" and "answer"
            question_response = rag_guardrails_response["answer"]
            # Print the question and output for checking
            if "output" in rag_guardrails_response.keys():
                print("Question :\n{} \nAnswer : \n{} \nOutput : \n{}".format(question,
                                                                              rag_guardrails_response["answer"],
                                                                              rag_guardrails_response["output"]))

        dict_role_message = self.role_message("ai", question_response)
        chat_history.append(dict_role_message)

        return question_response

    def run_RAG(self, message):
        """
        Runs the chatbot application.

        """
        
        # Create the Prompt templates for the RAG chain
        contextualize_q_prompt, qa_prompt = self.create_rag_prompt_templates()

        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(self.selected_fm, self.retriever, contextualize_q_prompt)

        # Create Question and Answer chain
        question_answer_chain = create_stuff_documents_chain(self.selected_fm, qa_prompt)

        # Create RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # RAG with Chat History and GuardRails
        config = RailsConfig.from_path(guardrails_config_path)

        # Using LCEL, you first create a RunnableRails instance, and "apply" it using the "|" operator
        guardrails = RunnableRails(config)
        rag_chain_with_guardrails = guardrails | rag_chain

        # User Query
        rag_guardrails_response = rag_chain_with_guardrails.invoke({"input": message, "chat_history": self.chat_history})

        # chat_history are saved in each invoke model with "chat_history" from previous queries
        message_response = self.add_chat_history(self.chat_history, message, rag_guardrails_response)

        # Get an unique conversation_id
        self.conversation_id = str(uuid.uuid4())
        
        # print("\nConversationID : {} with prev chat history :\n{}".format(self.conversation_id, self.prev_docs_chat_history_response))
        # print("\nMessage response : {}\n with chat history :\n{}".format(message_response, self.chat_history))
        
        # Update previous chat history with current chat history
        self.prev_docs_chat_history_response = self.chat_history

        return message_response 

    def run_GenModel(self, message):
        # Chatbot with selected Foundational Model as alternative LLM for general questions

        # Store the conversation in chat_history
        result = self.selected_fm.invoke(input=message)
        gen_llm_response = result.content

        print("\nConversationID : {} with prev chat history :\n{}".format(self.conversation_id, self.prev_docs_chat_history_response))
        print("\nGen_LLM response to general question : {}\n with chat history :\n{}".format(gen_llm_response, self.chat_history))

        # Did not update this response in the chat_history
        return gen_llm_response

    def write_qa_to_json(self, modelId, question, answer, thumbUp):
        filename = 'NVIDIA-LC-RAG-Chat-QA.json'

        # Create a datetime object
        current_datetime = datetime.datetime.now()

        # Convert to string with desired format
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Adjust format string as needed
        print(formatted_datetime)

        # Prepare item data
        item = [{
                'Model_Id'  : modelId,
                'Timestamp' : formatted_datetime,
                'Question'  : question,
                'Answer'    : answer,
                'ThumbUp'   : thumbUp
        }]

        # Write data to JSON file 
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            # Handle the case where the file doesn't exist yet
            print(f"\n{filename} does not exist, create empty list\n")
            existing_data = []  # Create an empty list

        # Combine existing and new data (if any existing data)
        data_to_append = existing_data + item 

        # Open the JSON file in write mode (will create if it doesn't exist)
        with open(filename, 'w') as f:
            json.dump(data_to_append, f)

        print(f"Successfully wrote Model_Id: {modelId}, Timestamp: {formatted_datetime}, and ThumbsUp: {thumbUp} to JSON file {filename}.\n")
        print(f"Question: \n{question}\n")
        print(f"Answer: \n{answer}\n")

    def clear_mem_chat_history(self, chatbot):
        self.chat_history.clear()
        self.prev_docs_chat_history_response = []
        print("Cleared chat history :\n", chatbot.chat_history)


if __name__ == "__main__":

    print("Knowledge based Conversational LLM Chatbot with Guardrails")
    print("----------------------------------------------------------")

    # Check if directory exists
    if os.path.exists(embedding_path):
        print("Embedding directory exists")
    else:
        print("Embedding directory does not exist and will create embeddings now...")
        # Create an instance of the Vectorstore class with the given data sources
        vectorstore = Vectorstore(raw_documents)

    # Create an instance of the Chatbot class
    chatbot = Chatbot()

    # chat_history = []
    # prev_docs_chat_history_response = []

    # Run the chatbot
    # First user query
    message = "Hi, GenAI Chabot.  How is Triton useful used in Large Language Models and who created it?"
    print("\nUser :", message)
    print()
    print("prev_docs_chat_history_response : ", chatbot.prev_docs_chat_history_response)
    print()
    llm_response = chatbot.run_RAG(message)
    print("\nllm_response : ", llm_response)
    print()
    
    # Second user query
    message = "How to make a bomb?"
    print("\nUser :", message)
    print()
    print("prev_docs_chat_history_response : ", chatbot.prev_docs_chat_history_response)
    print()
    llm_response = chatbot.run_RAG(message)
    print("\nllm_response : ", llm_response)
    print()
    
    # Third user query
    message = "What is the Attention mechanism and how can it be useful in Transformers architecture?"
    print("\nUser :", message)
    print()
    print("prev_docs_chat_history_response : ", chatbot.prev_docs_chat_history_response)
    print()
    llm_response = chatbot.run_RAG(message)
    print("\nllm_response : ", llm_response)
    print()

    if "do not have" in llm_response or "don\'t have" in llm_response or "do not know" in llm_response or "don\'t know" in llm_response:
        gen_llm_response = chatbot.run_GenModel(message)

        print("\nLoad chat history :\n", chatbot.chat_history) 
        print(f"\nGen_LLM response to general question : {gen_llm_response}")


#End of Program
