import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

def create_vector_store(csv_path: str):
    """
    Loads the FAQ dataset and creates a FAISS vector store using OpenAI embeddings.
    """
    print("Loading dataset...")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    documents = loader.load()
    
    print(f"Loaded {len(documents)} FAQs. Creating embeddings...")
    # Initialize OpenAI Embeddings (text-embedding-ada-002)
    embeddings = OpenAIEmbeddings()
    
    # Create the FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def setup_qa_chain(vector_store):
    """
    Sets up the QA chain using modern LCEL, the FAISS vector store, and OpenAI GPT-3.5-Turbo.
    """
    # Initialize the LLM (ChatGPT)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
    )
    
    # Define a custom chat prompt template for the chatbot
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a polite, helpful, and professional customer support assistant for Instacart.\nUse the following pieces of context to answer the user's question accurately.\nIf you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Build the chain using modern LCEL (LangChain Expression Language)
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain

def main():
    parser = argparse.ArgumentParser(description="Instacart FAQ Chatbot")
    parser.add_argument("--data", default="dataset.csv", help="Path to the FAQ CSV dataset")
    args = parser.parse_args()
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file containing 'OPENAI_API_KEY=sk-your_key_here' before running.")
        return

    # Check for Dataset
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset file '{args.data}' not found.")
        return

    try:
        # Initialize Vector Store and QA Chain
        vector_store = create_vector_store(args.data)
        qa_chain = setup_qa_chain(vector_store)
    except Exception as e:
        print(f"Failed to initialize the chatbot. Error: {e}")
        return
        
    print("\n" + "="*50)
    print("Welcome to the Instacart FAQ Chatbot!")
    print("Ask me anything about deliveries, Instacart+, orders, payments, or stores.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)
    
    # Interactive Chat Loop
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
                
            if not query.strip():
                continue
                
            # Invoke the QA chain (LCEL string input)
            response = qa_chain.invoke(query)
            
            print(f"\nInstacart Bot: {response.strip()}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred while generating the response: {e}")

if __name__ == "__main__":
    main()
