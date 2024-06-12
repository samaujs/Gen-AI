import boto3
import json
import streamlit as st
import uuid
import nvidia_lc
from datetime import datetime
import os

# Configure session state
HEAD_ICON = "images/RAG_bot_86x78.png"
# HEAD_ICON = "images/Nvidia_ChatRAG.png"
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"

# Section 1 : Initializations
if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
else:
    user_id = str(uuid.uuid4())
    st.session_state["user_id"] = user_id

if "llm_chain" not in st.session_state:

    print("Knowledge based Conversational LLM Chatbot with Guardrails")
    print("----------------------------------------------------------")

    # Check if directory exists
    if os.path.exists(nvidia_lc.embedding_path):
        print("Embedding directory exists")
    else:
        print("Embedding directory does not exist and will create embeddings now...")
        # Create an instance of the Vectorstore class with the given data sources
        vectorstore = nvidia_lc.Vectorstore(nvidia_lc.raw_documents)

    # Create an instance of the Chatbot class
    chatbot = nvidia_lc.Chatbot()

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Current date and time =", dt_string)

    st.session_state["llm_app"] = nvidia_lc
    st.session_state["llm_chain"] = chatbot

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input" not in st.session_state:
    st.session_state.input = ""

# Section 2 : Write Main Header
def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 2.3])
    with col1:
        st.image(HEAD_ICON, use_column_width="always")
    with col2:
        header = "NVIDIA-LC Conversational RAG Chatbot"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear

clear = write_top_bar()

if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""

    # Clear LLMChain Conversation Buffer Memory 
    st.session_state.llm_chain.clear_mem_chat_history(st.session_state["llm_chain"])

# Section 3 : Handling input from the user.
def handle_input():
    input = st.session_state.input

    # Handle User Query
    llm_chain = st.session_state["llm_chain"]
    llm_app = st.session_state["llm_app"]

    llm_response = llm_chain.run_RAG(message=input)
    if llm_response is not None:
        print("\nLLM Response for question :", llm_response)
    else:
        print("Input is None due to run out credits in NVIDIA!")
        # return

    # Handle the case for general questions that cannot be answered by RAG model with NVIDIA NeMo GuardRails
    if "do not have" in llm_response or "don\'t have" in llm_response or "do not know" in llm_response or "don\'t know" in llm_response :
        llm_response = llm_chain.run_GenModel(message=input)

        if llm_response is not None:
            print("\nLLM Response for general question :", llm_response)
        else:
            print("Input is None due to run out credits in NVIDIA!")
            # return

    # Change thumbUp based on user input in the future
    model = 'meta/llama3-8b-instruct'
    llm_chain.write_qa_to_json(modelId = model,
                               question = input,
                               answer = llm_response,
                               thumbUp = True)

    question_with_id = {
        "question": input,
        "id": len(st.session_state.questions) + 1,
    }
    st.session_state.questions.append(question_with_id)

    st.session_state.answers.append(
        {"answer": llm_response, "id": len(st.session_state.questions)}
    )
    st.session_state.input = ""

    print()
    print("st.session_state.questions :", st.session_state.questions)
    print("st.session_state.answers :", st.session_state.answers)
    print()

# Section 4 :
def write_user_message(md):
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(USER_ICON, use_column_width="always")
    with col2:
        st.warning(md["question"])
        # st.write(f"Tokens used: {md['tokens']}")

def render_answer(answer):
    col1, col2 = st.columns([1, 12])
    with col1:
        st.image(AI_ICON, use_column_width="always")
    with col2:
        # print("answer_response :", answer["response"])
        # st.info(answer["response"])
        st.info(answer)

def write_chat_message(md):
    chat = st.container()
    with chat:
        render_answer(md["answer"])

with st.container():
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        write_user_message(q)
        write_chat_message(a)

st.markdown("---")
input = st.text_input(
    "Hi, I am BC 2701, your AI Assistant with knowledge in Generative AI and Graphics Processing Unit (GPU). Please ask any question.", key="input", on_change=handle_input
)
