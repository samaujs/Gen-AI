import boto3
import json
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

import datetime
import time

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
        "url"   : "https://docs.cohere.com/docs/preambles"},
    {
        "title" : "What is prompt engineering?",
        "url"   : "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-prompt-engineering.html"},
    {
        "title" : "Query a knowledge base and generate responses based off the retrieved data",
        "url"   : "https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve-generate.html"},
    {
        "title" : "Improve the relevance of query responses with a reranker model in Amazon Bedrock",
        "url"   : "https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html"},
    {
        "title" : "AWS managed policies for Amazon Bedrock",
        "url"   : "https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam-awsmanpol.html"}
]

# Amazon Bedrock Knowledge Base (OpenSearch Serverless) ID
kbId = "7GX5TQSHLF"
guardrailId = 'i9w2hrcadw31'
numberOfResults = 20

# Prompt templates
system_prompt = "You are an AI assistant with expertise in Amazon Web Services.\
                 You should say you do not know if you do not know and answer only if you are very confident.\
                 Provide concise answer in numbered format and highlight api calls."

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
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

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

        # Creates bedrock_runtime
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

        # Creates bedrock_agent_runtime
        self.bedrock_agent_runtime_us_west = boto3.client(service_name='bedrock-agent-runtime', region_name='us-west-2')

        # Creates the ConversationBufferMemory for General Model
        self.memory_chat_history = ConversationBufferMemory(memory_key="chat_history")

        # Save the RAG chat history
        self.prev_docs_chat_history_response = []

        # Save the citations and cited_documents
        self.citations = []
        self.cited_documents = []

        # Save sessionId for multi-turn conversation in "retrieve_and_generate_stream"
        self.prev_sessionId = ''

        # Save "invocationMetrics" when using "invoke_model_with_response_stream" or "retrieve_and_generate_stream"
        # Need to compute the usage metrics when using "retrieve_and_generate_stream"
        self.inputTokenCount = 0
        self.outputTokenCount = 0
        self.est_totalTokenCount = 0
        self.latency = 0

    # Implement for Guardrails to overcome "TypeError: Object of type HumanMessage is not JSON serializable" with rag_chain
    def cohere_user_role_message(self, role: str, message: str):
        # This key pairing is specific to different FM model, "role" and "message" keys are tested for "cohere.command-r-plus-v1:0"
        return {"role": role, "message": message}

    def add_cohere_chat_history(self, chat_history, role_user, question, role_ai, question_response):
        # role : USER (Cohere), user (Claude), Human (LangChain)
        dict_role_message = self.cohere_user_role_message(role_user, question)
        chat_history.append(dict_role_message)

        # Might need to manoeurve with guardrail response compare code from Nvidia_llc
        # question_response = rag_guardrails_response["output"] or question_response = rag_guardrails_response["answer"]

        # role : CHATBOT (Cohere), assistant (Claude), AI (LangChain)
        dict_role_message = self.cohere_user_role_message(role_ai, question_response)
        chat_history.append(dict_role_message)

        return question_response

    def converse_user_role_message(self, role: str, message: str):
        # This key pairing is specific to Claude model with converse, "role" and "content" keys are tested for "us.amazon.nova-pro-v1:0"
        # {'role': 'user', 'content': [{'text': 'What shares should I buy?'}]}
        return {"role": role, 'content': [{'text': message}]}

    def add_converse_query(self, chat_history, role_user, message):
        dict_role_message = self.converse_user_role_message(role_user, message)
        chat_history.append(dict_role_message)
        # return chat_history is not necessary

    def claude_user_role_message(self, role: str, message: str):
        # This key pairing is specific to Claude model with converse, "role" and "content" keys are tested for "us.amazon.nova-pro-v1:0"
        # [{"role": "user", "content": "What shares should I buy?"}]
        return {"role": role, 'content': message}

    def add_claude_query(self, chat_history, role_user, message):
        dict_role_message = self.claude_user_role_message(role_user, message)
        chat_history.append(dict_role_message)
        # return chat_history is not necessary

    def add_claude_chat_history(self, chat_history, role_user, question, role_ai, question_response):
        # {"role": "user", "content": "Hello there."}
        dict_role_message = self.claude_user_role_message(role_user, question)
        chat_history.append(dict_role_message)

        # {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"}
        dict_role_message = self.claude_user_role_message(role_ai, question_response)
        chat_history.append(dict_role_message)

        return question_response

    def compute_token_usage(self, est_total_input_tokens, message, llm_response):
        chars_per_input_token = 2.5
        chars_per_output_token = 3.9
    
        # Estimate Input Tokens
        no_of_input_chars = len(message)
        est_inputTokenCount = est_total_input_tokens + int(no_of_input_chars/chars_per_input_token)

        # Estimate Output Tokens
        no_of_output_chars = len(llm_response)
        est_outputTokenCount = int(no_of_output_chars/chars_per_output_token)

        # Add previous total tokens with estimated input and output tokens
        est_total_input_tokens = est_inputTokenCount + est_outputTokenCount
        print(f"** Estimated total input tokens including previous total tokens : {est_total_input_tokens} **")

        return est_inputTokenCount, est_outputTokenCount, est_total_input_tokens

    # Obtain Cohere LLM response
    def run_Cohere_Model(self, message, chat_history):
        """
        Runs the streaming chatbot application with Cohere

        """

        # Initialise internal variables
        modelId = "cohere.command-r-plus-v1:0"
        contentType = "application/json"
        accept = "*/*"

        # Reset previous all needed variables before getting new LLM responses
        self.citations = []
        self.cited_documents = []
        self.inputTokenCount = 0
        self.outputTokenCount = 0
        self.latency = 0
        # citation_cnt = 1
        # document_cnt = 1

        print("Chat history :\n", chat_history)
        print()
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

            # Include Amazon Bedrock Guardrails (streaming responses are affected)
            streaming_response = self.bedrock_runtime.invoke_model_with_response_stream(body=cohere_body, modelId=modelId,
                                                                                        accept=accept, contentType=contentType,
                                                                                        # guardrailIdentifier = guardrailId,  # us-west-2
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

            # Include Amazon Bedrock Guardrails (streaming responses are affected)
            streaming_response = self.bedrock_runtime.invoke_model_with_response_stream(body=cohere_body, modelId=modelId,
                                                                                        accept=accept, contentType=contentType,
                                                                                        # guardrailIdentifier = guardrailId,  # us-west-2
                                                                                        # guardrailVersion ="1", 
                                                                                        # trace = "ENABLED"
                                                                                        )

        print(f"User message : {message}")
        print()
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
                message_response = self.add_cohere_chat_history(chat_history, "USER", message,
                                                                "CHATBOT", llm_stream_end_response)
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

                if 'amazon-bedrock-invocationMetrics' in chunk:
                    print(f"Invocation metrics: \n{chunk['amazon-bedrock-invocationMetrics']}")

                    self.inputTokenCount = chunk['amazon-bedrock-invocationMetrics']['inputTokenCount']
                    self.outputTokenCount = chunk['amazon-bedrock-invocationMetrics']['outputTokenCount']
                    self.latency = chunk['amazon-bedrock-invocationMetrics']['invocationLatency']
                elif 'amazon-bedrock-trace' in chunk:
                    # Useful only when guardrails are present
                    print(f"Amazon Bedrock Trace :\n{chunk['amazon-bedrock-trace']}")
                    print()
                    print(f"Denied topic :\n{chunk['amazon-bedrock-trace']['guardrail']['input'][guardrailId]['topicPolicy']['topics'][0]['name']}")

                print()
                print("ConversationID : {} with updated chat history :\n{}".format(self.conversation_id, chat_history))
        
                # Return newline
                return "\n"

    # Obtain Cohere LLM response
    def KB_Retrieve_and_Generate_Rerank_Stream(self, message, kbId, modelId, sessionId=None):
        region = 'us-west-2'
        rerank_modelId = "cohere.rerank-v3-5:0"

        # Workaround for Nova model exception error with modelArn
        if modelId == 'us.amazon.nova-pro-v1:0':
            model_package_arn = modelId
        elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
            model_package_arn = f"arn:aws:bedrock:{region}::foundation-model/{modelId}"

        rerank_model_package_arn = f"arn:aws:bedrock:{region}::foundation-model/{rerank_modelId}"

        # Prompt templates
        kb_resp_gen_prompt = f"""
                              Human: You are an AI assistant with expertise in Amazon Web Services and should refer to the retrieved context enclosed in <context> tags.

                              Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
                              If you don't know the answer, just say that you don't know, don't try to make up an answer.
                              <context>
                              $search_results$
                              </context>

                              <question>
                              $query$
                              </question>

                              The answer should be specific with supported citations and displayed in numbered bullet format.

                              Assistant:"""

        kb_orchestration_prompt = f"""
                                   Human: You are an AI assistant with expertise in Amazon Web Services and should orchestrate the retrieved context enclosed in <context> tags and take into considerations on previous chat history, $conversation_history$.

                                   Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
                                   If you don't know the answer, just say that you don't know, don't try to make up an answer.
                                   <context>
                                   $search_results$
                                   </context>

                                   <question>
                                   $query$
                                   </question>

                                   The response should be specific with supported citations and formatted according to $knowledge_base_guideline$ and $output_format_instructions$.\
                                   An example of citation format in reference :
                                   [<counter>number</counter>] <document>document file name</document> :: 👉 "<text>cited text in context</text>"

                                   Assistant:"""

        if sessionId:
            return self.bedrock_agent_runtime_us_west.retrieve_and_generate_stream(
                input={
                    'text': message
                },
                retrieveAndGenerateConfiguration={
                    'knowledgeBaseConfiguration': {
                        'generationConfiguration': {
                            'additionalModelRequestFields': {
                                'string': None
                            },
                            'guardrailConfiguration': {
                                'guardrailId': guardrailId,
                                'guardrailVersion': '1'
                            },
                            'inferenceConfig': {
                                'textInferenceConfig': {
                                    'maxTokens': 1000,
                                    'temperature': 0.1,
                                    'topP': 0.999
                                }
                            },
                            'performanceConfig': {
                                'latency': 'standard'
                            },
                            'promptTemplate': {
                                'textPromptTemplate': kb_orchestration_prompt
                            }
                        },
                        'knowledgeBaseId': kbId,
                        'modelArn': model_package_arn,
                        'orchestrationConfiguration': {
                            'additionalModelRequestFields': {
                                'string': None
                            },
                            'inferenceConfig': {
                                'textInferenceConfig': {
                                    'maxTokens': 1000,
                                    'temperature': 0.1,
                                    'topP': 0.999
                                }
                            },
                            'performanceConfig': {
                                'latency': 'standard'
                            },
                            'promptTemplate': {
                                'textPromptTemplate': kb_orchestration_prompt
                            }
                        },
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': numberOfResults,
                                'overrideSearchType': 'SEMANTIC',
                                'rerankingConfiguration': {
                                    'bedrockRerankingConfiguration': {
                                        'modelConfiguration': {
                                            'modelArn': rerank_model_package_arn
                                        },
                                        'numberOfRerankedResults': numberOfResults
                                    },
                                    'type': 'BEDROCK_RERANKING_MODEL'
                                }
                            }
                        }
                    },
                    'type': 'KNOWLEDGE_BASE'
                },
                # Value to maintain multi-turn interactions and contexts
                sessionId=sessionId
        )
        else:
            return self.bedrock_agent_runtime_us_west.retrieve_and_generate_stream(
                input={
                    'text': message
                },
                retrieveAndGenerateConfiguration={
                    'knowledgeBaseConfiguration': {
                        'generationConfiguration': {
                            'additionalModelRequestFields': {
                                'string': None
                            },
                            'guardrailConfiguration': {
                                'guardrailId': guardrailId,
                                'guardrailVersion': '1'
                            },
                            'inferenceConfig': {
                                'textInferenceConfig': {
                                    'maxTokens': 1000,
                                    'temperature': 0.1,
                                    'topP': 0.999
                                }
                            },
                            'performanceConfig': {
                                'latency': 'standard'
                            },
                            'promptTemplate': {
                                'textPromptTemplate': kb_orchestration_prompt
                            }
                        },
                        'knowledgeBaseId': kbId,
                        'modelArn': model_package_arn,
                        'orchestrationConfiguration': {
                            'additionalModelRequestFields': {
                                'string': None
                            },
                            'inferenceConfig': {
                                'textInferenceConfig': {
                                    'maxTokens': 1000,
                                    'temperature': 0.1,
                                    'topP': 0.999
                                }
                            },
                            'performanceConfig': {
                                'latency': 'standard'
                            },
                            'promptTemplate': {
                                'textPromptTemplate': kb_orchestration_prompt
                            }
                        },
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': numberOfResults,
                                'overrideSearchType': 'SEMANTIC',
                                'rerankingConfiguration': {
                                    'bedrockRerankingConfiguration': {
                                        'modelConfiguration': {
                                            'modelArn': rerank_model_package_arn
                                        },
                                        'numberOfRerankedResults': numberOfResults
                                    },
                                    'type': 'BEDROCK_RERANKING_MODEL'
                                }
                            }
                        }
                    },
                    'type': 'KNOWLEDGE_BASE'
                }
            )

    def run_Claude_Model(self, modelId, message, chat_history):
        """
        Runs the streaming chatbot application with Claude

        """

        # Initialise internal variables

        # Reset previous citations and cited_documents if present before getting new LLM responses
        self.citations = []
        self.cited_documents = []
        self.inputTokenCount = 0
        self.outputTokenCount = 0
        self.latency = 0

        print("Chat history :\n", chat_history)
        print()
        self.conversation_id = str(uuid.uuid4())

        chunks = ""
        no_of_citations = 1

        # Measure latency (ms)
        start_time = time.time()
        if self.prev_sessionId:
            print(f"Multi-turn conversation with prev_sessionId : {self.prev_sessionId}")
            retrieve_gen_response = self.KB_Retrieve_and_Generate_Rerank_Stream(message, kbId, modelId,
                                    sessionId=self.prev_sessionId)
        else:
            print(f"Start a new conversation should not have any prev_sessionId : {self.prev_sessionId}")
            retrieve_gen_response = self.KB_Retrieve_and_Generate_Rerank_Stream(message, kbId, modelId)
            if 'sessionId' not in retrieve_gen_response:
                print("There was no sessionId after being blocked from guardrail")
                retrieve_gen_response['sessionId'] = self.prev_sessionId
            self.prev_sessionId = retrieve_gen_response['sessionId']

            # For estimation when using "retrieve_and_generate", reset only for new session
            self.est_totalTokenCount = 0

        print(f"User message : {message}")
        print()
        for event in retrieve_gen_response['stream']:
            # event.keys() are 'output' and 'citation'
            print(f"Event : {event}")
            if 'output' in event:
                chunks += event['output']['text']
                yield event['output']["text"] or ""
                # print(event['output']['text'])
            elif 'citation' in event:

                for j in range(len(event['citation']['citation']['retrievedReferences'])):
                    print()
                    print(f"Retrieved references [{no_of_citations}] from OSS :\n")
                    print(f"{no_of_citations}.{j+1} Partial text response :\n")
                    print(event['citation']['citation']['generatedResponsePart']['textResponsePart']['text'])
                    print(100*"-")

                    print(f"{no_of_citations}.{j+1} Retrieved reference text :\n")
                    # Need to revise
                    self.citations.append(event['citation']['citation']['retrievedReferences'][j]['content']['text'])

                    print(event['citation']['citation']['retrievedReferences'][j]['content']['text'])
                    print(100*"-")

                    print(f"{no_of_citations}.{j+1} Retrieved reference URI :\n")
                    self.cited_documents.append(event['citation']['citation']['retrievedReferences'][j]['metadata']['x-amz-bedrock-kb-source-uri'])
                    print(event['citation']['citation']['retrievedReferences'][j]['metadata']['x-amz-bedrock-kb-source-uri'])
                    print(100*"-")

                print(100*"*")
                no_of_citations += 1

        # End of streaming response
        print()
        print("Chatbot stream-end response :\n", chunks)

        print()
        print("ConversationID : {} with prev chat history :\n{}".format(self.conversation_id, chat_history))
        # Update "chat_history" with complete response
        message_response = self.add_claude_chat_history(chat_history, "user", message,
                                                        "assistant", chunks)
        print()
        print("Message response :\n", message_response)

        # Compute estimated usage metrics
        print()
        print("Compute Usage Metrics :")
        self.inputTokenCount, self.outputTokenCount, self.est_totalTokenCount = self.compute_token_usage(self.est_totalTokenCount,
                                                                                                         message, message_response)

        self.latency = int((time.time() - start_time)*1000)

        print(f"Est. Input tokens : {self.inputTokenCount},\
                Est. Output tokens : {self.outputTokenCount},\
                Est. Total tokens : {self.est_totalTokenCount},\
                Est. Latency (Ms) : {self.latency}")

        print()
        print("ConversationID : {} with updated chat history :\n{}".format(self.conversation_id, chat_history))

        # Display the citations and source documents
        # citation_cnt, document_cnt = 1, 1
        # if self.citations:
        #     print("\n\nCITATIONS:")
        #     for citation in self.citations:
        #         print("[{}] {}".format(citation_cnt, citation))
        #         citation_cnt += 1

        # if self.cited_documents:
        #     print("\nDOCUMENTS:")
        #     for document in self.cited_documents:
        #         print("[{}] {}".format(document_cnt, document))
        #         document_cnt += 1

        # Return newline
        return "\n"

    def run_Claude_Model_native(self, modelId, message, chat_history):
        """
        Runs the streaming chatbot application with Claude

        """

        # Initialise internal variables

        # Reset previous citations and cited_documents if present before getting new LLM responses
        self.citations = []
        self.cited_documents = []
        self.inputTokenCount = 0
        self.outputTokenCount = 0
        self.latency = 0

        print("Chat history :\n", chat_history)
        print()
        self.conversation_id = str(uuid.uuid4())

        chunks = ""
        no_of_citations = 1

        # Add query in chat_history for multi-turn conversation
        self.add_claude_query(chat_history, "user", message)

        print("Multi-turn chat history should ends with a question :\n", chat_history)
        print()

        # Format the request payload using the model's native structure.
        native_request = {
             "anthropic_version": "bedrock-2023-05-31",
             "max_tokens": 1000,
             "system": system_prompt,
             "messages": chat_history,
             "temperature": 0.1,
             "top_p": 0.999,
             "top_k": 250,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        # Invoke the model with the request.
        llm_invoke_stream_response = self.bedrock_runtime.invoke_model_with_response_stream(modelId=modelId, body=request)

        print(f"User message : {message}")
        print()

        # Process streaming response by returning chunks to streamlit application
        for event in llm_invoke_stream_response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            # print("chunk :", chunk)

            if chunk["type"] == "content_block_delta":
                # print(chunk['delta']['text'], end="|", flush=True)
                chunks += chunk["delta"]["text"]
                yield chunk["delta"]["text"] or ""

        # End of streaming response
        print()
        print("Chatbot stream-end response :\n", chunks)

        print()
        print("ConversationID : {} with prev chat history :\n{}".format(self.conversation_id, chat_history))

        # Update "chat_history" with complete response to be used for multi-turn conversation
        self.add_claude_query(chat_history, "assistant", chunks)

        print()
        print("Message response :\n", chunks)

        # Capture metrics from the last chunk
        if 'amazon-bedrock-invocationMetrics' in chunk:
            print(f"amazon-bedrock-invocationMetrics: \n{chunk['amazon-bedrock-invocationMetrics']}")
            self.inputTokenCount = chunk['amazon-bedrock-invocationMetrics']['inputTokenCount']
            self.outputTokenCount = chunk['amazon-bedrock-invocationMetrics']['outputTokenCount']
            self.latency = chunk['amazon-bedrock-invocationMetrics']['invocationLatency']

        print()
        print("ConversationID : {} with updated chat history :\n{}".format(self.conversation_id, chat_history))

        return "\n"

    def KB_Retrieve(self, query, kbId, numberOfResults=5):
        return self.bedrock_agent_runtime_us_west.retrieve(
                                    retrievalQuery= {
                                                     'text': query
                                                    },
                                    knowledgeBaseId=kbId,
                                    retrievalConfiguration= {
                                                             'vectorSearchConfiguration': {
                                                                                           'numberOfResults': numberOfResults,
                                                                                           'overrideSearchType': "HYBRID", # 'SEMANTIC'
                                                                                          }
                                                            }
                                   )

    # Fetch context from the retrieval results
    def get_contexts(self, retrievalResults):
        contexts = []
        for retrievedResult in retrievalResults: 
            contexts.append(retrievedResult['content']['text'])
        return contexts

    def run_Nova_Model(self, modelId, message, chat_history):
        """
        Runs the streaming chatbot application with Nova

        """

        # Initialise internal variables
        self.conversation_id = str(uuid.uuid4())
        chunks = ""

        # Prompt templates
        # system_prompt = "You are an AI assistant with expertise in Amazon Web Services."\
        #                 "You should say you do not know if you do not know and answer only if you are very confident."\
        #                 "Provide answer in number bulleted format."

        # Define your system prompt for Nova
        system = [{ "text": system_prompt}]

        # Configure the inference parameters.
        inf_params = {"temperature": 0.1, "topP": 0.999, "maxTokens": 1000}

        # messages format with role = "user" or "assistant"
        # messages = [ {"role": "user", "content": [{"text": message}]}, 
        #              {"role": "assistant", "content": [{"text": llm_response["output"]["message"]["content"][0]["text"]}]},
        #              {"role": "user", "content": [{"text": "Give an example how to write this IAM policy"}]},]

        # Reset previous all needed variables before getting new LLM responses
        self.inputTokenCount = 0
        self.outputTokenCount = 0
        self.latency = 0

        # Note : Ensure chat_history must be of messages format above
        print("Chat history :\n", chat_history)
        print()

        print(f"User message : {message}")
        print()

        # Retrieve embeddings closest to the provided "message"  from Amazon Bedrock Knowledge Base
        retrieve_response = self.KB_Retrieve(message, kbId, numberOfResults)
        retrievalResults = retrieve_response['retrievalResults']
        contexts = self.get_contexts(retrievalResults)

        context_prompt = f"""
                Human: You are an AI assistant with expertise in Amazon Web Services and provides answers to questions by using fact based \
                on context information when possible. Use the following pieces of information to provide a concise answer to the question \
                enclosed in <question> tags.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                <context>
                {contexts}
                </context>

                <question>
                {message}
                </question>

                The response should be specific and use facts when possible. Provide answer in number bulleted format.

                Assistant:"""

        if not chat_history:
            # Beginning of multi-turn conversation
            # chat_history = [{ "role":'user', "content": [{'text': context_prompt}]}]
            self.add_converse_query(chat_history, "user", context_prompt)
        else:
            # Add query in chat_history for multi-turn conversation
            self.add_converse_query(chat_history, "user", message)

        print("Multi-turn chat history should ends with a question :\n", chat_history)
        print()

        # Include Amazon Bedrock Guardrails (streaming responses are not affected with 'async', 'Prompt Attack' with in-context prompt))
        llm_converse_response = self.bedrock_runtime.converse_stream(modelId=modelId, messages=chat_history,
                                                                     system=system, inferenceConfig=inf_params,
                                                                     # guardrailConfig={
                                                               #               'guardrailIdentifier': guardrailId,  # us-west-2
                                                               #               'guardrailVersion': '1',
                                                               #               'trace': 'enabled',
                                                               #               'streamProcessingMode': 'async'  # advise to use sync mode
                                                               #       },
                                                                     )

        llm_converse_stream_response = llm_converse_response.get("stream")
        # Process streaming response by returning chunks to streamlit application
        if llm_converse_stream_response:
            for event in llm_converse_stream_response:
                # print(f"Event : {event}")
                if "contentBlockDelta" in event:
                    chunks += event["contentBlockDelta"]["delta"]["text"]
                    yield event['contentBlockDelta']["delta"]["text"] or ""

        # End of streaming response
        print()
        print("Chatbot stream-end response :\n", chunks)

        print()
        print("ConversationID : {} with prev chat history :\n{}".format(self.conversation_id, chat_history))

        # Update "chat_history" with complete response to be used for multi-turn conversation
        self.add_converse_query(chat_history, "assistant", chunks)

        print()
        print("Message response :\n", chunks)

        if 'metadata' in event:
            print(f"Metadata: \n{event['metadata']}")
            self.inputTokenCount = event['metadata']['usage']['inputTokens']
            self.outputTokenCount = event['metadata']['usage']['outputTokens']
            self.latency = event['metadata']['metrics']['latencyMs']

        elif event['metadata']['usage']['totalTokens'] == 0:
            # Useful only when guardrails are present
            print(f"Deny topic :\n{event['metadata']['trace']['guardrail']['inputAssessment'][guardrailId]['topicPolicy']['topics']}")
            print()
            print(f"Content policy filter :\n{event['metadata']['trace']['guardrail']['inputAssessment'][guardrailId]['contentPolicy']['filters'][0]['type']}")

        print()
        print("ConversationID : {} with updated chat history :\n{}".format(self.conversation_id, chat_history))

        return "\n"

    def run_GenModel(self, message):
        modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
        anthropic_version = 'bedrock-2023-05-31'

        # ChatBedrock with Claude3 Sonnet as alternative LLM for general questions
        claude3_sonnet_llm = ChatBedrock(
                                    client=self.bedrock_runtime,
                                    model_id=modelId,
                                    region_name="us-west-2",
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

    def write_qa_to_json(self, modelId, question, answer, thumbUp):
        filename = 'LLMChat-RAG-Rerank-QA.json'

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

    def write_qa_to_dynamodb(self, modelId, question, answer, thumbUp):
        # Configure boto3 client (replace with your credentials and region)
        dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
        table = dynamodb.Table('GenAI-Rerank-RAG-Chat')

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

        chatbot.prev_sessionId = ''
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
