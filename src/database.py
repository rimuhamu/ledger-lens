import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class Database:
    def __init__(self, db_path="./data/vectorstore"):
        self.db_path = db_path
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()

    def ingest_document(self, file_path):
        """
        Ingest a PDF document into the vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not file_path or not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return False

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                print(f"Error: No documents found in {file_path}")
                return False

            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                print(f"Error: No chunks created from {file_path}")
                return False

            # Create or update the vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.db_path
                )
            else:
                self.vectorstore.add_documents(chunks)

            print(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            print(f"Error ingesting document: {e}")
            return False

    def get_retriever(self, k: int = 3):
        """
        Get the retriever from the vector store.
        Loads from disk if vectorstore not initialized but exists.
        
        Args:
            k: Number of documents to retrieve (default: 3)
            
        Returns:
            Retriever object or None if no vectorstore exists
        """        
        if self.vectorstore is None:
            if os.path.exists(self.db_path) and os.listdir(self.db_path):
                try:
                    self.vectorstore = Chroma(
                        persist_directory=self.db_path,
                        embedding_function=self.embeddings
                    )
                except Exception as e:
                    print(f"Error loading vector store: {e}")
                    return None
            else:
                print("Warning: No vector store found. Ingest documents first.")
                return None

        return self.vectorstore.as_retriever(search_kwargs={"k": k})