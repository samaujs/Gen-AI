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

# Using Streamlit to build the User interface
from streamlit_feedback import streamlit_feedback

# Configure session state
HEAD_ICON = "images/aws-lol.png"
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"

# Has to be first streamlit call
st.set_page_config(page_title="Enterprise Chatbot", page_icon=HEAD_ICON)

# Application variables
cohere_model_native_title = "Cohere R+ (Hnswlib)"
claude_model_native_title = "Claude 3.5 Sonnet (native)"
claude_model_title = "Claude 3.5 Sonnet (KB+GR+Rerank)"
nova_model_title = "Nova Pro (latency)"

prev_docs_chat_history = []

# Section 1 : Write Main Header
def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 3])
    with col1:
        st.image(HEAD_ICON, use_column_width="always")
    with col2:
        header = "Amazon Bedrock Enterprise Chatbot"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear

clear = write_top_bar()

# Section 2 : Select different foundational models
st.divider()

# Added to reset radio button when clear is true
# Initialize default selection flag if not present
if "model_selection" not in st.session_state:
    st.session_state.model_selection = claude_model_native_title
#else:
#    print(f"Select radio button with prev_model : {st.session_state.prev_model}")
#    st.session_state.model_selection = st.session_state.prev_model

llm_select_model = st.radio(
    "Select the Foundational Model: \n\n",
    [claude_model_native_title, claude_model_title, nova_model_title],
    key="llm_select_model",

    # Added to reset radio button when clear is true
    index=[claude_model_native_title, claude_model_title, nova_model_title].index(st.session_state.model_selection),

    horizontal=True,
)

#st.session_state.prev_model = llm_select_model
#st.session_state.model_selection = st.session_state.prev_model
st.session_state.model_selection = llm_select_model

# Update model_selection when radio changes
print(f"Selected radio button : {llm_select_model}")

if llm_select_model == claude_model_native_title:
    modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
elif llm_select_model == claude_model_title:
    modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
elif llm_select_model == nova_model_title:
    modelId = 'us.amazon.nova-pro-v1:0'
else:
    modelId = 'cohere.command-r-plus-v1:0'

print(f"In Section 2, with selected LLM modelID : \"{modelId}\".")

# Section 3 : Initializations of all state variables
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

# Added for feedback
if "prev_model" not in st.session_state:
    st.session_state.prev_model = ""

if "answer_fbk" not in st.session_state:
    st.session_state.answer_fbk = []

if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

# Reset all state variables
if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""

    # Clear LLMChain Conversation Buffer Memory 
    st.session_state.llm_chain.clear_mem_chat_history(st.session_state["llm_chain"])
    prev_docs_chat_history = []
    st.session_state.chat_history = []

    # Reset radio button
    # del st.session_state.llm_select_model # claude_model_native_title
    # Set flag to use default model on next render
    st.session_state.model_selection = claude_model_native_title
    print(f"Current model selection after reset : {st.session_state.model_selection}")

    # Added for feedback
    st.session_state.prev_model = ""
    st.session_state.answer_fbk = []
    # Does not reload the streamlit_feedback widget, need to reload webpage to work
    st.session_state.feedback_key = 0

# Section 4 : Handling input from the user.
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

# Section 5 : Display all the Questions and Answers
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

# with st.container():
#     for q, a in zip(st.session_state.questions, st.session_state.answers):
#         write_user_message(q)
#         write_chat_message(a)

# Section 6 : Collecting answer feedback from user
def write_user_feedback(answer_fbk_score):
    """
    Update the answer feedback history.

    The Questions and Answers are already saved in st.session_state.questions and st.session_state.answers
    """

    st.session_state.answer_fbk.append(
        {"answer_fbk": answer_fbk_score, "id": len(st.session_state.answer_fbk) + 1}
    )
    print()
    print("Stored user feedback from responses :", st.session_state.answer_fbk)

def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}")
    print(f"Feedback submitted: {user_response}, {emoji}")

    write_user_feedback(answer_fbk_score=user_response['score'])

    # Set new feedback key for next session
    st.session_state.feedback_key += 1

    # Save Questions and Answers with user feedback
    question = st.session_state.questions[-1]['question']
    answer = st.session_state.answers[-1]['answer']
    thumbUp = True
    
    if user_response['score'] == '👎':
        thumbUp = False

    llm_chain = st.session_state["llm_chain"]

    # Write to JSON formatted output file
    llm_chain.write_qa_to_json(modelId = modelId,
                               question = question,
                               answer = answer,
                               thumbUp = thumbUp)

    # Write to DynamoDB table
    llm_chain.write_qa_to_dynamodb(modelId = modelId,
                                   question = question,
                                   answer = answer,
                                   thumbUp = thumbUp)

def get_thumbs_count():
    data = st.session_state.answer_fbk

    # Count the number of entries with 'answer_fbk' equal to ''
    count_thumbs_up = sum(item['answer_fbk'] == '👍' for item in data)
    count_thumbs_down = sum(item['answer_fbk'] == '👎' for item in data)

    return count_thumbs_up, count_thumbs_down


st.markdown("---")
# input = st.text_input(
    # "Hi, I am your AI Assistant with knowledge in Large Language Models. Please ask any question.", key="input", on_change=handle_input
# )

# Display chat_history conversation
print("Print history conversation")
print(100*"=")

for message in st.session_state.chat_history:
    # Cohere model with "invoke_model_with_response_stream" API
    if llm_select_model == cohere_model_native_title:  # nova_model_title
        # Cohere model with "invoke_model_with_response_stream" API
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

    # For Claude model with "retrieve_and_generate_stream" or "invoke_model_with_response_stream" API
    # claude_model_native_title, claude_model_title
    else:
        if message['role'] == 'user':
            # For Nova model with "converse_stream" API
            # message['content'][0]['text']
            print("Query : {}".format(message['content']))
            print()
            with st.chat_message("Human", avatar=USER_ICON):
                # st.markdown(message['content'][0]['text'])
                st.markdown(message['content'])
        elif message['role'] == 'assistant':
            # message['content'][0]['text'] for converse api
            print("LLM response : {}".format(message['content']))
            print(100*"-")
            with st.chat_message("AI", avatar=AI_ICON):
                # st.markdown(message['content'][0]['text'])
                st.markdown(message['content'])

# Obtain user query
user_query = st.chat_input("Hi, I am your AI Assistant with knowledge in Amazon Bedrock. Please ask your question.")

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

        print("Begin to work on a LLM response...")
        print("st.session_state.chat_history :\n", prev_docs_chat_history)

        # Perform streaming and extracted information based on the foundational model
        if llm_select_model == claude_model_native_title:
            print(f"\n** Selected model title : {llm_select_model} **")

            # For handling streaming response
            streaming_response = st.write_stream(llm_chain.run_Claude_Model_native(modelId, user_query, prev_docs_chat_history))
            print()
            print("Application complete streaming_response from Claude model (** no RAG **) :\n", streaming_response)
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
                document_cnt = 1

                # Store only the non-duplicated text with the information for first occurrence
                for i in range(len(llm_chain.cited_documents)):
                    if llm_chain.cited_documents[i]['text'] not in doc_refs_text:
                        doc_refs_text.append(llm_chain.cited_documents[i]['text'])
                        doc_refs.append({'id': llm_chain.cited_documents[i]['id'], 'text': llm_chain.cited_documents[i]['text'],
                                         'title' : llm_chain.cited_documents[i]['title'], 'url' : llm_chain.cited_documents[i]['url']})

                print("\nREFERENCES :")
                for document in doc_refs:
                    print(f"[{document_cnt}] {document}")
                    mention(label=f"[{document_cnt}] {document['title']} :: 👉 \"{document['text']}\"", url=document['url'])
                    document_cnt += 1

            # Display token usage if present
            print()
            print(f"** Input tokens in app : {llm_chain.inputTokenCount} **")
            print(f"** Output tokens in app : {llm_chain.outputTokenCount} **")
            print(f"** Invocation Latency in app : {llm_chain.latency} **")
            print()

            if llm_chain.inputTokenCount > 0 and llm_chain.outputTokenCount > 0:
                st.write(f"** Usage Metrics - Input Tokens : {llm_chain.inputTokenCount}, \
                                              Output Tokens : {llm_chain.outputTokenCount} and \
                                              Invocation Latency : {llm_chain.latency} **")
           
        elif llm_select_model == claude_model_title:
            print(f"\n** Selected model title : {llm_select_model} **")

            # For handling streaming response
            streaming_response = st.write_stream(llm_chain.run_Claude_Model(modelId, user_query, prev_docs_chat_history))
            print()
            print("Application complete streaming_response from Claude model :\n", streaming_response)
            print()

            # For clearer display as separator (not blocked by guardrail) using "retrieve_and_generate_stream"
            if prev_docs_chat_history[-1]['content'] != "Sorry, I am unable to assist you with this request.":
                st.write("** REFERENCES : **")

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
                document_cnt = 1

                # Store only the non-duplicated text with the information for first occurrence
                for i in range(len(llm_chain.cited_documents)):
                    if llm_chain.citations[i] not in doc_refs_text:
                        doc_refs_text.append(llm_chain.citations[i])
                        doc_refs.append({'text': llm_chain.citations[i],
                                         'x-amz-bedrock-kb-source-uri' : llm_chain.cited_documents[i]})

                print("\nREFERENCES :")
                for document in doc_refs:
                    print(f"[{document_cnt}] {document}")

                    # Extract filename from URI path
                    ref_str = document['x-amz-bedrock-kb-source-uri']
                    ref_filename = ref_str.split("/")[-1]
                    mention(label=f"[{document_cnt}] {ref_filename} :: 👉 \"{document['text']}\"",
                                     url=document['x-amz-bedrock-kb-source-uri'])
                    document_cnt += 1

            # Display token usage if present using "converse_stream"
            print()
            print(f"** Est. Input tokens in app : {llm_chain.inputTokenCount} **")
            print(f"** Est. Output tokens in app : {llm_chain.outputTokenCount} **")
            print(f"** Est. Latency (Ms) in app : {llm_chain.latency} **")
            print()

            # "retrieve_and_generate" does not return usage metrics, so compute respective estimates
            if llm_chain.inputTokenCount > 0 and llm_chain.outputTokenCount > 0:
                st.write(f"** Usage Metrics - Est. Input Tokens : {llm_chain.inputTokenCount}, \
                                              Est. Output Tokens : {llm_chain.outputTokenCount} and \
                                              Est. Latency (Ms) : {llm_chain.latency} **")
           
        elif llm_select_model == nova_model_title:
            print(f"\n** Selected model title : {llm_select_model} **")

            # For handling streaming response
            # streaming_response = st.write_stream(llm_chain.run_Nova_Model(modelId, user_query, prev_docs_chat_history))
            # Call run_Claude_Model with Nova Pro modelId
            streaming_response = st.write_stream(llm_chain.run_Claude_Model(modelId, user_query, prev_docs_chat_history))
            print()
            print("Application complete streaming_response from Nova model :\n", streaming_response)
            print()

            # For clearer display as separator (not blocked by guardrail) using "retrieve_and_generate_stream"
            if prev_docs_chat_history[-1]['content'] != "Sorry, I am unable to assist you with this request.":
                st.write("** REFERENCES : **")

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
                document_cnt = 1

                # Store only the non-duplicated text with the information for first occurrence
                for i in range(len(llm_chain.cited_documents)):
                    if llm_chain.citations[i] not in doc_refs_text:
                        doc_refs_text.append(llm_chain.citations[i])
                        doc_refs.append({'text': llm_chain.citations[i],
                                         'x-amz-bedrock-kb-source-uri' : llm_chain.cited_documents[i]})

                print("\nREFERENCES :")
                for document in doc_refs:
                    print(f"[{document_cnt}] {document}")

                    # Extract filename from URI path
                    ref_str = document['x-amz-bedrock-kb-source-uri']
                    ref_filename = ref_str.split("/")[-1]
                    mention(label=f"[{document_cnt}] {ref_filename} :: 👉 \"{document['text']}\"",
                                     url=document['x-amz-bedrock-kb-source-uri'])
                    document_cnt += 1

            # Display token usage if present using "converse_stream"
            print()
            print(f"** Est. Input tokens in app : {llm_chain.inputTokenCount} **")
            print(f"** Est. Output tokens in app : {llm_chain.outputTokenCount} **")
            print(f"** Est. Latency (Ms) in app : {llm_chain.latency} **")
            print()

            # "retrieve_and_generate" does not return usage metrics, so compute respective estimates
            if llm_chain.inputTokenCount > 0 and llm_chain.outputTokenCount > 0:
                st.write(f"** Usage Metrics - Est. Input Tokens : {llm_chain.inputTokenCount}, \
                                              Est. Output Tokens : {llm_chain.outputTokenCount} and \
                                              Est. Latency (Ms) : {llm_chain.latency} **")
           
           
        print()
        print("Updated prev_docs_chat_history :\n", prev_docs_chat_history)

        # Added for feedback
        question_with_id = {
            "question": user_query,
            "id": len(st.session_state.questions) + 1,
        }
        st.session_state.questions.append(question_with_id)

        st.session_state.answers.append(
            {"answer": streaming_response, "id": len(st.session_state.questions)}
        )
        st.session_state.input = ""

        print()
        print("st.session_state.questions :", st.session_state.questions)
        print("st.session_state.answers :", st.session_state.answers)
        print()

# Added for feedback
# if st.session_state.prev_model != modelId:
if st.session_state.prev_model != llm_select_model:
    # For different model, displays the last user feedback on LLM response
    feedback_key = f"ans_feedback_{st.session_state.feedback_key}"
else:
    # For same model, increments the next uniquely identifier by 1 for the widget component
    feedback_key = f"ans_feedback_{st.session_state.feedback_key + 1}" 

# Set the prev_model
# st.session_state.prev_model = modelId
st.session_state.prev_model = llm_select_model

print()
print("^^ Updated prev_model ^^ :\n", st.session_state.prev_model)
print(f"feedback_key : {feedback_key}")

# Use the streamlit_feedback library to obtain user feedback on the generated LLM response
if len(st.session_state.answers) > 0:
    answer_fbk = streamlit_feedback(
                                     feedback_type="thumbs",
                                     align="center",  # "flex-start",
                                     key=feedback_key,
                                     on_submit=_submit_feedback
                                    )
    print(f"LLM answer feedback of {feedback_key} : {answer_fbk}\n")

    if answer_fbk is not None:
        count_thumbs_up, count_thumbs_down = get_thumbs_count()
        print(f"Number of '👍' : {count_thumbs_up} and Number of '👎' : {count_thumbs_down}")
        print()
        st.write(f"Number of '👍' : {count_thumbs_up} and Number of '👎' : {count_thumbs_down}")

print("\n** End of Program : {} **".format(user_query))
