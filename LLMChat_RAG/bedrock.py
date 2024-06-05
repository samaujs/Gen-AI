import boto3
import json
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

import datetime

# LangChain with ConversationBufferMemory
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Documents that are to be embedded and store in Vector Database
raw_documents = [
    {
        "title": "Text Embeddings",
        "url": "https://docs.cohere.com/docs/text-embeddings"},
    {
        "title": "Similarity Between Words and Sentences",
        "url": "https://docs.cohere.com/docs/similarity-between-words-and-sentences"},
    {
        "title": "The Attention Mechanism",
        "url": "https://docs.cohere.com/docs/the-attention-mechanism"},
    {
        "title": "Transformer Models",
        "url": "https://docs.cohere.com/docs/transformer-models"}
]

class Vectorstore:
    """
    A class representing a collection of documents indexed into a vectorstore.

    Parameters:
    raw_documents (list): A list of dictionaries representing the sources of the raw documents. Each dictionary should have 'title' and 'url' keys.

    Attributes:
    raw_documents (list): A list of dictionaries representing the raw documents.
    docs (list): A list of dictionaries representing the chunked documents, with 'title', 'text', and 'url' keys.
    docs_embs (list): A list of the associated embeddings for the document chunks.
    docs_len (int): The number of document chunks in the collection.
    idx (hnswlib.Index): The index used for document retrieval.

    Methods:
    load_and_chunk(): Loads the data from the sources and partitions the HTML content into chunks.
    embed(): Embeds the document chunks using the Cohere API.
    index(): Indexes the document chunks for efficient retrieval.
    retrieve(): Retrieves document chunks based on the given query.
    """

    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 5  # 10; top_k retrieved documents

        # Specifies the region for bedrock_runtime
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

        self.modelId = 'cohere.embed-english-v3'
        self.contentType = 'application/json'
        self.accept = '*/*'
        
        self.load_and_chunk()
        self.embed()
        self.index()


    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("1. Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """

        print("2. Embedding document chunks...")
        num_embed_called = 1
        
        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            
            print("Docs_len : {}, Embed counter : {}".format(len(self.docs), num_embed_called))
            cohere_body = json.dumps({
                        "texts": texts,
                        "input_type": "search_document"  # search_query | classification  |clustering
                    })
            response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=self.modelId,
                                                         accept=self.accept, contentType=self.contentType)
            embed_response_body = json.loads(response.get('body').read())
            docs_embs_batch = embed_response_body.get('embeddings')
            num_embed_called += 1

            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the document chunks for efficient retrieval.
        """
        print("3. Indexing document chunks...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} document chunks.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """
        
        cohere_body = json.dumps({
            "texts": [query],
            "input_type": "search_query"  # search_document | classification  |clustering
        })
        response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=self.modelId,
                                                     accept=self.accept, contentType=self.contentType)
        embed_response_body = json.loads(response.get('body').read())
        query_emb = embed_response_body.get('embeddings')

        # Search is performed by the knn_query() method from the hnswlib library
        # Returns the document chunks most similar to the query.
        # Define the number of document chunks to return using the attribute self.retrieve_top_k=10
        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Optional : Reranking with Cohere "rerank-english-v2.0" endpoint
        
        docs_retrieved = []
        for doc_id in doc_ids:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )
    
        return docs_retrieved

# Single prompt at a time for streamlit LLM app
class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        """
        Initializes an instance of the Chatbot class.

        Parameters:
        vectorstore (Vectorstore): An instance of the Vectorstore class.

        """
        self.vectorstore = vectorstore
        self.conversation_id = "Begin..."

        # Specifies the region for bedrock_runtime
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

        # Creates the ConversationBufferMemory for General Model
        self.memory_chat_history = ConversationBufferMemory(memory_key="chat_history")

        # Save the RAG chat history
        self.prev_docs_chat_history_response = []


    def run_RAG(self, message):
        """
        Runs the chatbot application.

        """
        
        contentType = "application/json"
        accept = "*/*"

        # Generate search queries (if any) from user query
        modelId = "cohere.command-r-v1:0"            
        cohere_body = json.dumps({
                "temperature": 0.0,
                "p": 0.99,
                "k": 250,
                "max_tokens": 1000,
            
                "preamble": "You are BC, an AI assistant model 2701 and you are created by SAM. Your expertise is in Transformers and Attention models. \
                             You should say you do not know if you do not know and answer only if \
                             you are very confident. Organise the answers in a nice number bulleted format.", 
                # "chat_history" is not used for "search_queries_only" with empty []
                "message": message,
                "search_queries_only": True,
        })
        response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=modelId,
                                                     accept=accept, contentType=contentType)
        search_response_body = json.loads(response.get('body').read())

        # Use Cohere Command R+ for continuous chat after query search
        modelId = "cohere.command-r-plus-v1:0"
        # If there are search queries, retrieve document chunks and respond
        if search_response_body["search_queries"]:
            print("Retrieving information...\n", end="")

            # Retrieve document chunks for each query
            documents = []
            for query in search_response_body["search_queries"]:
                documents.extend(self.vectorstore.retrieve(query["text"]))

            # Use document chunks to respond
            cohere_body = json.dumps({
                    "temperature": 0.0,
                    "p": 0.99,
                    "k": 250,
                    "max_tokens": 1000,
                
                    "preamble": "You are BC, an AI assistant model 2701 and you are created by SAM. Your expertise is in Transformers and Attention models. \
                                 You should say you do not know if you do not know and answer only if \
                                 you are very confident. Organise the answers in a nice number bulleted format.", 
                    
                    "chat_history": self.prev_docs_chat_history_response,
                    "message": message,
                    "documents": documents,
            })
            response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=modelId,
                                                         accept=accept, contentType=contentType)
            docs_response_body = json.loads(response.get('body').read())

        # If there is no search query, directly respond
        else:
            cohere_body = json.dumps({
                    "temperature": 0.0,
                    "p": 0.99,
                    "k": 250,
                    "max_tokens": 1000,
                
                    "preamble": "You are BC, an AI assistant model 2701 and you are created by SAM. Your expertise is in Transformers and Attention models. \
                                 You should say you do not know if you do not know and answer only if \
                                 you are very confident. Organise the answers in a nice number bulleted format. Answer in a happy and cheerful tone.", 
                    
                    "chat_history": self.prev_docs_chat_history_response,
                    "message": message,
            })
            response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=modelId,
                                                         accept=accept, contentType=contentType)
            docs_response_body = json.loads(response.get('body').read())

        # Print the chatbot response, citations, and documents
        docs_response_text = docs_response_body.get('text')
        print("User:\n", message)
        print("Chatbot:\n", docs_response_text)
        citations = []
        cited_documents = []
                
        self.conversation_id = str(uuid.uuid4())
        
        print("\nConversationID : {} with prev chat history :\n{}".format(self.conversation_id, self.prev_docs_chat_history_response))
        docs_chat_history_response = docs_response_body.get('chat_history')

        # chat_history are saved in each invoke model with "chat_history" from previous queries
        self.prev_docs_chat_history_response = docs_chat_history_response
        print("\nConversationID : {} with chat history :\n{}".format(self.conversation_id, docs_chat_history_response))
        
        return docs_response_text, docs_chat_history_response

    def run_GenModel(self, message):
        modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
        anthropic_version = 'bedrock-2023-05-31'

        # ChatBedrock with Claude3 Sonnet as alternative LLM for general questions
        claude3_sonnet_llm = ChatBedrock(
                                    client=self.bedrock_runtime,
                                    model_id=modelId,
                                    region_name="us-east-1",
                                    model_kwargs={"temperature": 0.1,
                                                  "top_p": 0.999,
                                                  "top_k": 250,
                    
                                                  "anthropic_version": anthropic_version,
                                                  "max_tokens": 1000,
                                                 },
                            )

        prompt_template = """You are BC, an AI assistant model 2701 and you are created by SAM. Your expertise is tourism in ASEAN region and premium vacation planning.
                             You should say you do not know if you do not know and answer only if you are very confident.
                             Organise the answers in a nice number bulleted format. Answer in a happy and cheerful tone.

                             Previous conversation:
                             {chat_history}

                             Human: {input}
                             AI assistant:\n"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Using LLMChain for conversation and store the conversation in buffer memory
        LLMChain_conversation = LLMChain(
                                            llm=claude3_sonnet_llm,
                                            prompt=prompt,
                                            memory=self.memory_chat_history,
                                            verbose=True
                                )

        print("\nLoad conversation memory :\n", self.memory_chat_history.load_memory_variables({}))
        gen_llm_response = LLMChain_conversation.predict(input=message)
        print("\nLLM response to general question :\n", gen_llm_response)

        return gen_llm_response, self.memory_chat_history

    def write_qa_to_dynamodb(self, modelId, question, answer, thumbUp):
        # Configure boto3 client (replace with your credentials and region)
        dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
        table = dynamodb.Table('GenAI-RAG-Chat-QA')

        # Create a datetime object
        current_datetime = datetime.datetime.now()

        # Convert to string with desired format
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Adjust format string as needed
        print(formatted_datetime)

        # Prepare item data
        item = {
                'Model_Id'  : modelId,
                'Timestamp' : formatted_datetime,
                'Question'  : question,
                'Answer'    : answer,
                'ThumbUp'   : thumbUp
        }

        # Write data to DynamoDB
        table.put_item(Item=item)

        print(f"Successfully wrote Model_Id: {modelId}, Timestamp: {formatted_datetime}, and ThumbsUp: {thumbUp} to DynamoDB table.\n")
        print(f"Question: \n{question}\n")
        print(f"Answer: \n{answer}\n")

    def clear_mem_chat_history(self, chatbot):
        chatbot.memory_chat_history.clear()
        chatbot.prev_docs_chat_history_response = []
        print("Cleared LLMChain Conversation Buffer Memory :\n", chatbot.memory_chat_history.load_memory_variables({}))


if __name__ == "__main__":

    print("Amazon Bedrock RAG Chatbot")
    print("--------------------------")

    # Create an instance of the Vectorstore class with the given data sources
    vectorstore = Vectorstore(raw_documents)

    # Create an instance of the Chatbot class
    chatbot = Chatbot(vectorstore)

    # First user query
    message = "Hi, GenAI Chabot.  How is the transformer mechanism used in Large Language Models?"
    prev_docs_chat_history_response = []

    # Run the chatbot
    print("\nUser :", message)
    llm_response, prev_docs_chat_history_response = chatbot.run_RAG(message)
    print("llm_response : ", llm_response)
    print()
    print("prev_docs_chat_history_response : ", prev_docs_chat_history_response)
    print()
    
    # Second user query
    message = "What is the latest price of Nvidia share and when was that?  I would also like to have the price in the last 5 years?"
    print("\nUser :", message)
    llm_response, prev_docs_chat_history_response = chatbot.run_RAG(message)
    print("llm_response : ", llm_response)
    print()
    print("prev_docs_chat_history_response : ", prev_docs_chat_history_response)
    print()

    if "do not have" in llm_response or "don\'t have" in llm_response or "don\'t know" in llm_response:
        gen_llm_response, memory_chat_history = chatbot.run_GenModel(message)
        print("\nLoad conversation memory :\n", memory_chat_history.load_memory_variables({}))

#End of Program
