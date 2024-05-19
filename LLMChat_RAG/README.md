## Build Retrieval Augmented Generation (RAG) based Generative AI (GenAI) application with Amazon Bedrock
### The following describes the involved steps : 
> - Create AWS client for making inference requests
  - Create a boto3 client to connect programmatically for making inference requests for Foundational Models (FM) hosted in Amazon Bedrock (eg. Cohere Command R)

> - Explore Cohere Models with Embeddings
  - Load and Chunk (html documents with unstructured library).
  - Embed the document chunks in batches using Cohere Embed (English) model hosted in Amazon Bedrock.
  - Uses the hsnwlib package to index the document chunk embeddings.  This ensures efficient similarity search during retrieval.  \
    For simplicity, we use hsnwlib as vector library for our knowledge database.
  - Chatbot decides if it needs to consult external information from knowledge database before responding.  If so, it determines an optimal \
    set of search queries to use for documents retrieval.
  - The document search is performed by the knn_query() method from the hnswlib library.  With a user query message, it returns the document \
    chunks that are most similar to this query.  We can define the number of document chunks to return using the attribute retrieve_top_k().  \
    If there are matched documents, the retrieved document chunks are then passed as documents in a new query send to the FM (Cohere Command R+).
  - Display external information from RAG with citations and retrieved document chunks based on the retrieve_top_k parameter.
  - The chat history is updated for next user query.

> - Alternative LLM model to handle general questions
  - Using an alternative Foundational Models (FM) hosted in Amazon Bedrock (eg. Claude3 Sonnet) to handle general questions whenever the \
    Cohere models do not know the answers.  LangChain LLMChain and ConversationBufferMemory are used in to establish the conversation and \
    store the chat history.

> - LLM Chatbot with RAG User interface
  - Using Streamlit to build the User interface for the LLM Chatbot with RAG.
![alt text](https://github.com/samaujs/Gen-AI/images/Amazon_Bedrock_RAG_Chatbot_LOL.png?raw=true)

## Run the code :
```
> streamlit run app.py --server.port <port_number>
> http://<ip_addr>:<port_number>/ on browser
```

## References :<br>
>[1] [AWS Bedrock Generative AI Application Architecture](https://community.aws/content/2f2d59922DQNz3iH1pCTeudpmhv/aws-bedrock-generative-ai-application-architecture)<br>
>[2] [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)<br>
>[3] [Anthropic Claude 3](https://docs.anthropic.com/claude/docs/models-overview)<br>
>[4] [How to Build a RAG-Powered Chatbot with Chat, Embed, and Rerank](https://cohere.com/blog/rag-chatbot#embed-the-document-chunks)<br>
