import os
# exec(os.getenv("CODE"))   # to execute the whole code in huggingface.

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64
from io import BytesIO

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## going to each and very pdf and each page of that padf and extracting text from it.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

## converting chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding =embeddings)
    vector_store.save_local("faiss_index")

## developing bot
def get_conversational_chain():
    prompt_template= """ANALAYZE THE PDF CONTEXT and
    Answer the question as detailed as possible from the provided context, make sure to provide
    all the details if the answer is not in the provided context just say, "answer is not available in the context",
    don't provide the wrong answer.
    Context: \n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature= 0.9)
    prompt= PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt= prompt)
    return chain

## the user input interface
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization= True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()

    response= chain({"input_documents":docs, "question":user_question}, return_only_outputs=True)

    print(response)
    st.write("Bot: ", response["output_text"])

# streamlit app
def main():
    st.set_page_config(page_title="Chat With Multiple PDF")

    # Function to set a background image
    def set_background(image_file):
        with open(image_file, "rb") as image:
            b64_image = base64.b64encode(image.read()).decode("utf-8")
        css = f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{b64_image});
            background-size: cover;
            background-position: centre;
            backgroun-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    # Set the background image
    set_background("assets/background_image.png")

    st.header("Gemini-Decode : PDF Bot 💬📄")

    user_question = st.text_input("Ask any Question from the PDF Files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        st.image("assets/sidebar_image.png", use_column_width=True)
        pdf_docs = st.file_uploader("Upload Your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type='pdf')
        if st.button("Submit & Process") :
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        st.image("assets/sidebar_image1.png", use_column_width=True)

if __name__ == "__main__":
    main()
