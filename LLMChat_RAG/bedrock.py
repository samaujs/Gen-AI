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
        "title" : "Crafting Effective Prompts",
        "url"   : "https://docs.cohere.com/docs/crafting-effective-prompts"},
    {
        "title" : "Advanced Prompt Engineering Techniques",
        "url"   : "https://docs.cohere.com/docs/advanced-prompt-engineering-techniques"},
    {
        "title" : "Prompt Truncation",
        "url"   : "https://docs.cohere.com/docs/prompt-truncation"},
    {
        "title" : "System Messages - Preambles",
        "url"   : "https://docs.cohere.com/docs/preambles"}
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
        
        # Load and chunk data, embed document chunks and index the document chunks for efficient retrieval
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

        # Save the citations and cited_documents
        self.citations = []
        self.cited_documents = []


    # Coded should be placed in bedrock.py after streaming works
    # Implement for Guardrails to overcome "TypeError: Object of type HumanMessage is not JSON serializable" with rag_chain
    def role_message(self, role: str, message: str):
        # This key pairing is specific to different FM model, "role" and "message" keys are tested for "cohere.command-r-plus-v1:0"
        return {"role": role, "message": message}

    def add_chat_history(self, chat_history, question, question_response):
        dict_role_message = self.role_message("USER", question)  # Human
        chat_history.append(dict_role_message)

        # Might need to manoeurve with guardrail response compare code from Nvidia_llc
        # question_response = rag_guardrails_response["output"] or question_response = rag_guardrails_response["answer"]

        dict_role_message = self.role_message("CHATBOT", question_response)  # AI
        chat_history.append(dict_role_message)

        return question_response

    # Obtain LLM response
    def get_LLM_response(self, message, chat_history):
        """
        Runs the streaming chatbot application.

        """

        # Initialise internal variables
        modelId = "cohere.command-r-plus-v1:0"
        contentType = "application/json"
        accept = "*/*"

        # Reset previous citations and cited_documents if present before getting new LLM responses
        self.citations = []
        self.cited_documents = []
        # citation_cnt = 1
        # document_cnt = 1

        print("Chat history :\n", chat_history)
        self.conversation_id = str(uuid.uuid4())

        # Generate search queries (if any) from user query
        cohere_body = json.dumps({
                                  "temperature": 0.0,
                                  "p": 0.99,
                                  "k": 250,
                                  "max_tokens": 1000,
   
                                  # "preamble" not needed for seach queries
                                  # "chat_history": chat_history is not used for "search_queries_only" with empty []
                                  "message": message,
                                  "search_queries_only": True, # is used only to search for embeddings matching the user query with no output
                                 })

        response = self.bedrock_runtime.invoke_model(body=cohere_body, modelId=modelId,
                                                     accept=accept, contentType=contentType)
 
        # Use only for "invoke_model", non-streaming for searching the relevant documents based on the user query
        search_response_body = json.loads(response.get('body').read())

        # If there are search queries, retrieve document chunks and respond
        if search_response_body["search_queries"]:
            print()
            print("Search queries are present will be used to retrieve top_k documents...\n", end="")

            # Retrieve document chunks for each search query
            documents = []
            for query in search_response_body["search_queries"]:
                doc_chunk = self.vectorstore.retrieve(query["text"])
                print(f"Query from search_queries :\n{query}\n with document chunk information :\n{doc_chunk}")
                print()

                # append documents with the retrieved document chunks
                documents.extend(doc_chunk)

            # Retrieve information from Amazon Knowledge Base OpenSearch Serveless collections
            # documents = self.vectorstore.retrieve(query)
            # print(f"Query : {query} with no. of retrieved documents : {len(documents)}")
            # for query in search_response_body["search_queries"]:
            #     documents.extend(self.vectorstore.retrieve(query["text"]))

            print(f"Query : {message} with no. of retrieved documents : {len(documents)}")
            print(f"Retrieved document chunks to be used as context for LLM :\n{documents}")
            print()

            # Use document chunks to respond
            cohere_body = json.dumps({
                                      "temperature": 0.0,
                                      "p": 0.99,
                                      "k": 250,
                                      "max_tokens": 1000,
    
                                      "preamble": "You are an AI assistant with expertise in Large Language Models provided by Cohere. \
                                                   Answer the question considering the history of the conversations. \
                                                   You should say you do not know if you do not know and answer only if \
                                                   you are very confident. Organise the answers in a nice number bulleted format.",
                                      "chat_history": chat_history,
                                      "message": message,
                                      "documents": documents # used as context from the search_queries for LLM
                                     })

            # Include Amazon Bedrock Guardrails
            streaming_response = self.bedrock_runtime.invoke_model_with_response_stream(body=cohere_body, modelId=modelId,
                                                                                        accept=accept, contentType=contentType,
                                                                                        # guardrailIdentifier = '1ozptvv2saiw',
                                                                                        # guardrailVersion ="1", 
                                                                                        # trace = "ENABLED"
                                                                                        )

            # For streaming, change "invoke_model" to "invoke_model_with_response_stream"
            # For non-streaming with "invoke_model" only
            # Print the chatbot response and chat_history with docs_response_body['chat_history']
            # docs_response_text = docs_response_body.get('text')

            # chat_history are saved in each invoke model with "chat_history" from previous queries
            # message_response = add_chat_history(chat_history, message, docs_response_text)

        # If there is no search queries result, directly respond without "documents"
        else:
            print("\n** Call LLM without additional context from RAG **")
            cohere_body = json.dumps({
                                      "temperature": 0.0,
                                      "p": 0.99,
                                      "k": 250,
                                      "max_tokens": 1000,
    
                                      "preamble": "You are an AI assistant with expertise in Large Language Models provided by Cohere. \
                                                   Answer the question considering the history of the conversations. \
                                                   You should say you do not know if you do not know and answer only if \
                                                   you are very confident. Organise the answers in a nice number bulleted format.",
                                      "chat_history": chat_history,
                                      "message": message,
                                     })

            # Include Amazon Bedrock Guardrails
            streaming_response = self.bedrock_runtime.invoke_model_with_response_stream(body=cohere_body, modelId=modelId,
                                                                                        accept=accept, contentType=contentType,
                                                                                        # guardrailIdentifier = '1ozptvv2saiw',
                                                                                        # guardrailVersion ="1", 
                                                                                        # trace = "ENABLED"
                                                                                        )

        # Same processing whether there are "search_queries"
        # Process streaming response by returning chunks to streamlit application
        for event in streaming_response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            # print("chunk :", chunk)
            if chunk["event_type"] == "text-generation":
                yield chunk["text"] or ""
            elif chunk["event_type"] == "stream-end":
                # End of streaming response
                llm_stream_end_response = chunk['response']['text']
                print()
                print("Chatbot stream-end response :\n", llm_stream_end_response)

                # Store the citations and cited_documents
                if 'citations' in chunk['response'] and 'documents' in chunk['response'] :
                    self.citations = chunk['response']['citations']
                    self.cited_documents = chunk['response']['documents']
                else:
                    print(100*"=")
                    print("Chunk response :\n", chunk['response'])
                    print(100*"=")

                # docs_chat_history_response = chunk['response']['chat_history']

                print()
                print("ConversationID : {} with prev chat history :\n{}".format(self.conversation_id, chat_history))
                # Update "chat_history" with complete response
                message_response = self.add_chat_history(chat_history, message, llm_stream_end_response)
                print()
                print("Message response :\n", message_response)
                print()
                # Display the citations and source documents
                # if self.citations:
                #    print("\n\nCITATIONS:")
                #    for citation in self.citations:
                #        print("[{}] {}".format(citation_cnt, citation))
                #        citation_cnt += 1

                # if self.cited_documents:
                #     print("\nDOCUMENTS:")
                #     for document in self.cited_documents:
                #         print("[{}] {}".format(document_cnt, document))
                #         document_cnt += 1

                print()
                print("ConversationID : {} with updated chat history :\n{}".format(self.conversation_id, chat_history))
        
                # Return newline
                return "\n"

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

        prompt_template = """You are an AI assistant model 0910 and created by RLA. Your expertise is Large Language Models.
                             You should say you do not know if you do not know and answer only if you are very confident.
                             Organise the answers in a nice number bulleted format. Answer in a Machine Learning professional tone.

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

# End of Program
