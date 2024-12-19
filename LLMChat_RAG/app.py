import boto3
import json
import streamlit as st
import uuid
import bedrock
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Added for citation with source in chat response
from streamlit_extras.mention import mention

# Configure session state
HEAD_ICON = "images/aws-lol.png"
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"

# Section 1 : Initializations
if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
else:
    user_id = str(uuid.uuid4())
    st.session_state["user_id"] = user_id

if "llm_chain" not in st.session_state:
    # Create an instance of the Vectorstore class with the given data sources
    vectorstore = bedrock.Vectorstore(bedrock.raw_documents)

    # Create an instance of the Chatbot class
    chatbot = bedrock.Chatbot(vectorstore)

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Current date and time =", dt_string)

    st.session_state["llm_app"] = bedrock
    st.session_state["llm_chain"] = chatbot

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input" not in st.session_state:
    st.session_state.input = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Added for streaming
st.set_page_config(page_title="Enterprise Chatbot", page_icon=HEAD_ICON)
prev_docs_chat_history = []


# Section 2 : Write Main Header
def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 3])
    with col1:
        st.image(HEAD_ICON, use_column_width="always")
    with col2:
        header = "Amazon Bedrock RAG Chatbot"
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
    prev_docs_chat_history = []
    st.session_state.chat_history = []

# Section 3 : Handling input from the user.
def handle_input():
    input = st.session_state.input

    # For testing
    print("User query :", input)
    llm_response = "No answer from LLM yet!"
    #st.session_state.chat_history.append(HumanMessage(input))
    # with st.chat_message("Human"):
    st.markdown(user_query)
    # with st.chat_message("AI"):
    #st.markdown("This is a default test response")

    # Handle User Query
    # llm_chain = st.session_state["llm_chain"]
    # llm_app = st.session_state["llm_app"]

    # llm_response, prev_docs_chat_history = llm_chain.run_RAG(input)
    modelId = 'cohere.command-r-plus-v1:0'

    # Handle the case for general questions that cannot be answered by Cohere RAG model
    # if "do not have" in llm_response or "don\'t have" in llm_response or "do not know" in llm_response or "don\'t know" in llm_response :
    #     llm_response, memory_chat_history = llm_chain.run_GenModel(input)
    #     modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

    # Change thumbUp based on user input in the future
    # llm_chain.write_qa_to_dynamodb(modelId = modelId,
    #                                question = input,
    #                                answer = llm_response,
    #                                thumbUp = True)

    question_with_id = {
        "question": input,
        "id": len(st.session_state.questions),
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
        # st.info(answer)
        st.write(answer)  # write_stream(answer)

def write_chat_message(md):
    chat = st.container()
    with chat:
        render_answer(md["answer"])

with st.container():
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        write_user_message(q)
        write_chat_message(a)


st.markdown("---")
# input = st.text_input(
    # "Hi, I am your AI Assistant with knowledge in Large Language Models. Please ask any question.", key="input", on_change=handle_input
# )

# Display chat_history conversation
for message in st.session_state.chat_history:
    if message['role'] == 'USER':
        print("Query : {}".format(message['message']))
        print()
        with st.chat_message("Human", avatar=USER_ICON):
            st.markdown(message['message'])
    elif message['role'] == 'CHATBOT':
        print("LLM response : {}".format(message['message']))
        print(100*"-")
        with st.chat_message("AI", avatar=AI_ICON):
            st.markdown(message['message'])

# Obtain user query
user_query = st.chat_input("Hi, I am your AI Assistant with knowledge in Large Language Models. Please ask any question.")

# Initialise the variables
llm_chain = st.session_state["llm_chain"]
llm_app = st.session_state["llm_app"]

if user_query is not None and user_query != "":
    with st.chat_message("Human", avatar=USER_ICON):
        st.markdown(user_query)

    with st.chat_message("AI", avatar=AI_ICON):
        # Cannot use "st.session_state.chat_history" directly because of "HumanMessage is not JSON serializable" format error
        # Try to save in the LLM specific format (eg. "role": role, "message": message)

        prev_docs_chat_history = st.session_state.chat_history
        print("st.session_state.chat_history :\n", prev_docs_chat_history)

        # Use only for "invoke_model", non-streaming
        # llm_response = llm_chain.get_LLM_response(user_query, prev_docs_chat_history)
        # st.markdown(llm_response)

        # For handling streaming response
        streaming_response = st.write_stream(llm_chain.get_LLM_response(user_query, prev_docs_chat_history))
        print()
        print("Application complete streaming_response :\n", streaming_response)
        print()
        print("-- If any citations after completion of streaming_response --")
        # Display the citations and source documents
        if llm_chain.citations:
            citation_cnt = 1
            print("CITATIONS:")
            for citation in llm_chain.citations:
                print("[{}] {}".format(citation_cnt, citation))
                citation_cnt += 1
        print()
        print("-- If any documents after completion of streaming_response --")
        if llm_chain.cited_documents:
            doc_refs_text = []
            doc_refs = []

            # Store only the non-duplicated text with the information for first occurrence
            for i in range(len(llm_chain.cited_documents)):
                if llm_chain.cited_documents[i]['text'] not in doc_refs_text:
                    doc_refs_text.append(llm_chain.cited_documents[i]['text'])
                    doc_refs.append({'id': llm_chain.cited_documents[i]['id'], 'text': llm_chain.cited_documents[i]['text'],
                                     'title' : llm_chain.cited_documents[i]['title'], 'url' : llm_chain.cited_documents[i]['url']})

            document_cnt = 1
            print("\nREFERENCES :")
            for document in doc_refs:
                print(f"[{document_cnt}] {document}")
                mention(label=f"[{document_cnt}] {document['title']} : {document['text']}", url=document['url'])
                document_cnt += 1

        print()
        print("Updated prev_docs_chat_history :\n", prev_docs_chat_history)

print("\n** End of Program : {} **".format(user_query))
