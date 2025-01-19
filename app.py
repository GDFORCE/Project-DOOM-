import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import json

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        st.write("---")
        for message in st.session_state.chat_history:
            if message.type == "human":
                st.markdown(f"<div style='text-align:right;'>{message.content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; color:blue;'>{message.content}</div>", unsafe_allow_html=True)
        st.write("---")
    else:
        st.error("Please process the PDFs first!")


def export_chat_history():
    if st.session_state.chat_history:
        history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history])
        st.download_button(
            label="Download Chat History",
            data=history,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    else:
        st.warning("No chat history available to download.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Project Doom,Your Personal AI ", page_icon=":books:", layout="wide")
    st.header("Project Doom :books:")

  
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = None
            st.success("Chat history cleared!")
    with col2:
        export_chat_history()

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        st.write(f"Number of text chunks created: {len(text_chunks)}")

                    
                        vectorstore = get_vectorstore(text_chunks)

                        
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []  
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No text extracted from the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF to process.")

        
        if st.session_state.conversation:
            st.success("Documents are ready for queries!")
        else:
            st.warning("No documents processed yet.")

if __name__ == '__main__':
    main()
