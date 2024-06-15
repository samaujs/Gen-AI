## Build a Knowledge-based Conversational LLM Chatbot with NVIDIA AI Foundation Endpoints, NeMo Guardrails and LangChain
### The following describes the involved steps :
- Create NVIDIA chat model for making inference requests via NVIDIA endpoint APIs
>   - Create an NVIDIA chat model with access to hosted NVIDIA Inference Microservices (NIM) of Foundational Models (eg. Meta Llama3, MistralAI Mixtral, Google Gemma)

- Explore Facebook AI Similarity Search (FAISS) with NVIDIA Embedding model
>   - Load and Chunk (html documents with Beautiful Soup library).
>   - Embed the document chunks using NVIDIAEmbeddings("NV-Embed-QA") model hosted in NVIDIA.
>   - Use the FAISS to index the document chunk embeddings and ensures efficient similarity search during retrieval.
>     The FAISS performs Similarity Search based on L2 distance and generated embeddings are stored locally the "embed" folder.

- Conversational LLM Chatbot with NVIDIA AI Foundation Endpoints, NeMo Guardrails and LangChain
>   - Build Conversational chatbot with Chat history using LangChain Framework (eg. Prompts, Foundation Models).
>   - Large Language Model (LLM) with NVIDIA  Nvidia Inference Microservice (NIM) API (NV-Embed-QA, Meta Llama3).
>   - User selects the Foundation Model (Meta Llama3-8b, MistralAI Mixtral-8x7b and Google Gemma-7b) for Question and Answers task.
>   - Craft prompts based on Context, Question and Chat history.
>   - Implement Retrieval Augmented Generation (RAG) and Vector store with FAISS.
>   - LangChain Conversational Chain, Document Chain and Retrieval Chain are used in to establish the user conversation.
>   - Add NeMo Guardrails (Input, Retrieval, Dialog, Output) to a RAG Chain.
>   - Chatbot with NeMo Guardrails (OpenAI gpt3.5) uses knowledge database and checks on input/output the before responding.
>   - The Chat history is updated to be used for next user query.
>   - Save Questions and Answers to a JSON file for evaluations and fine-tunings.

- Alternative LLM model to handle general questions
>   - Using an alternative Foundational Model (FM) hosted in NVIDIA (eg. Meta Llama3) to handle general questions whenever the
>     RAG Chain model do not know the answer.

- LLM Chatbot with RAG User interface
>   - Using Streamlit to build the User interface for the Conversational LLM Chatbot with RAG.
><br>

![alt text](https://github.com/samaujs/Gen-AI/blob/main/NVIDIA_LC_RAG/images/NVIDIA_LC_RAG_Chatbot_BC_2Q.png)

## Run the code :
```
> streamlit run app.py --server.port <port_number>
> http://<ip_addr>:<port_number>/ on browser
```

## References :<br>
>[1] [LangChain](https://python.langchain.com/v0.1/docs/get_started/introduction)<br>
>[2] [Nvidia AI Foundation Endpoints](https://python.langchain.com/v0.1/docs/integrations/providers/nvidia/#nvidia-ai-foundation-endpoints)<br>
>[3] [Nvidia NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)<br>
>[4] [Nvidia Embedding models](https://python.langchain.com/v0.1/docs/integrations/text_embedding/nvidia_ai_endpoints/)<br>
>[5] [Build Enterprise Retrieval-Augmented Generation Apps with NVIDIA Retrieval QA Embedding Model](https://developer.nvidia.com/blog/build-enterprise-retrieval-augmented-generation-apps-with-nvidia-retrieval-qa-embedding-model/)<br>
