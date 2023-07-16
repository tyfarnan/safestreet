import streamlit as st
import openai
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from IPython.display import display, Markdown
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
# Set up OpenAI API credentials

def run_llm_query(qa,query):
    # Call your Python-LLM function here and return the result
    result = qa.run(query)
    return result
def generate_embeddings():
    loader = DirectoryLoader('short_texts', glob="*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    embeddings = OpenAIEmbeddings()
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    return db
    #db.save_local("embed.db")
# Streamlit app
def main():
    st.title("Safestreet-sdXai")
    openai_api_key = "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"
    os.environ['OPENAI_API_KEY'] = openai_api_key
    embeddings = OpenAIEmbeddings()
    #db1=FAISS.load_local('embed.db',embeddings)
    db1=generate_embeddings()
    retriever = db1.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    
    # Input field for the query
    query = st.text_input("Safestreet can answer your questions regarding the safety of a neighbourhood. Ask Safestreet a question: ")
    
    # Run the query and display the result
    if st.button("Ask Safestreet"):
        # Call the function to run the LLM query
        result = run_llm_query(qa,query)
        
        # Display the result
        st.text("Safestreet Speaks:")
        st.write(result)

if __name__ == "__main__":
    main()
