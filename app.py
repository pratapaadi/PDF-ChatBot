import streamlit as st
import os
import google.generativeai as genai
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(text):
    text_spliltter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_spliltter.split_text(text)
    return chunks

def get_vector(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details,if the answer is not in the provided context
    just say,"answer is not available in the context",don't provide the wrong answer.
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:

    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()

    response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)
    print(response)
    st.write("Reply:",response["output_text"])

def main():
    st.set_page_config("Chat with multiple PDFs")
    st.header("Chat with Multiple PDFs using Gemini")
    user_question=st.text_input("Ask me anything about the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Your PDFs")
        pdf_docs=st.file_uploader("Upload PDF files and Click on Submit",type=["pdf"],accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Extracting Information"):
                    raw_text=get_pdf_text(pdf_docs)
                    text_chunks=get_chunks(raw_text)
                    get_vector(text_chunks)
                    st.success("DONE")
            else:
                st.warning("Please upload at least one PDF File!!")

if __name__=="__main__":
    main()


    
