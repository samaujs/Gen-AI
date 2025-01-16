## Build Retrieval Augmented Generation (RAG) based Generative AI (GenAI) application with Amazon Bedrock services
### The following describes the involved steps :
- AWS boto3 clients
>   - Create a 'bedrock_agent_runtime' client for making Amazon Knowledge Base retrieval requests
>   - Create a 'bedrock-runtime' client for making LLM inference requests

- 1. Cohere Command R+
>   - Simple web scraping to form RAG with Hnswlib vector store (In-Memory)
>   - Multi-turn conversation
>   - Streaming
>   - Citations
>   - Token Usage, Invocation Latency, First Byte Latency

- 2. Claude 3.5 Sonnet
>   - Amazon Bedrock Knowledge Base (Cohere Embed - English)
>   - Amazon OpenSearch Serverless (vector store, FAISS index + BM25)
>   - Documents (pdf) in S3
>   - Rerank (Cohere Rerank 3.5)
>   - Multi-turn conversation
>   - Streaming
>   - Citations
>   - Guardrails

- 3. Nova Pro
>   - Amazon Bedrock Knowledge Base (Cohere Embed - English)
>   - Amazon OpenSearch Serverless (vector store, FAISS index + BM25)
>   - Documents (pdf) in S3
>   - Multi-turn conversation
>   - Streaming
>   - In-context prompting
>   - Token Usage, Latency (fastest)

- LLM Chatbot with RAG User interface
>   - Using Streamlit to build the User interface for the LLM Chatbot with RAG
><br>

## Run the code :
```
> streamlit run app.py --server.port <port_number>
> http://<ip_addr>:<port_number>/ on browser
```

## References :<br>
>[1] [AWS Bedrock Generative AI Application Architecture](https://community.aws/content/2f2d59922DQNz3iH1pCTeudpmhv/aws-bedrock-generative-ai-application-architecture)<br>
>[2] [Build Retrieval Augmented Generation (RAG) based Generative AI (GenAI) application with Amazon Bedrock](https://community.aws/content/2f38mlpgBYSMBBGeUJNMjyUmRyw/build-retrieval-augmented-generation-rag-based-generative-ai-genai-application-with-amazon-bedrock)<br>
>[3] [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)<br>
>[4] [Anthropic Claude 3](https://docs.anthropic.com/claude/docs/models-overview)<br>
>[5] [How to Build a RAG-Powered Chatbot with Chat, Embed, and Rerank](https://cohere.com/blog/rag-chatbot#embed-the-document-chunks)<br>
