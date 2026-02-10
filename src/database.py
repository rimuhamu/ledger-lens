import os
import hashlib
import logging
from typing import Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Suppress noisy httpx HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class Database:
    """
    Database class managing both vector storage (Pinecone) and file storage (S3).
    
    Features:
    - Pinecone for semantic search with metadata filtering
    - S3 for persistent PDF storage
    - User-scoped document management
    """
    
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ledgerlens")
        
        # Create index if it doesn't exist
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL")  # For MinIO compatibility
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "ledgerlens-documents")
        
        # Create S3 bucket if it doesn't exist
        self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' already exists")
        except ClientError:
            try:
                region = os.getenv("AWS_REGION", "us-east-1")
                if region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"Created S3 bucket: {self.bucket_name}")
            except ClientError as e:
                logger.error(f"Failed to create bucket: {e}")
    
    def _generate_doc_id(self, user_id: str, ticker: str, filename: str):
        """Generate unique document ID from user, ticker, and filename"""
        unique_string = f"{user_id}:{ticker}:{filename}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def _upload_to_s3(self, file_path: str, s3_key: str):
        """
        Upload file to S3 and return the S3 key.
        
        Args:
            file_path: Local path to file
            s3_key: S3 object key (path in bucket)
            
        Returns:
            S3 key of uploaded file
        """
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def _download_from_s3(self, s3_key: str, local_path: str):
        """
        Download file from S3 to local path.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded from S3: {s3_key} -> {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def ingest_document(
        self, 
        file_path: str, 
        user_id: str, 
        ticker: str,
        filename: str,
    ):
        """
        Ingest a PDF document into Pinecone and S3.
        
        Args:
            file_path: Path to the PDF file
            user_id: User identifier
            ticker: Company ticker symbol
            filename: Original filename
            
        Returns:
            Dict with document_id, num_chunks, num_pages, s3_key
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        document_id = self._generate_doc_id(user_id, ticker, filename)
        
        # Upload PDF to S3
        s3_key = f"users/{user_id}/documents/{ticker}/{document_id}_{filename}"
        self._upload_to_s3(file_path, s3_key)
        
        try:
            # Load and chunk the document
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No content extracted from PDF")
            
            # Generate all embeddings in batch
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}#{idx}"
                metadata = {
                    "document_id": document_id,
                    "user_id": user_id,
                    "ticker": ticker.upper(),
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk.page_content,
                    "page": chunk.metadata.get("page", 0),
                    "s3_key": s3_key,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"✓ Ingested {len(chunks)} chunks for document {document_id}")
            
            return {
                "document_id": document_id,
                "num_chunks": len(chunks),
                "num_pages": len(documents),
                "s3_key": s3_key,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            # Cleanup S3 if ingestion failed
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            except:
                pass
            raise e
    
    def query_documents(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        ticker: Optional[str] = None,
        top_k: int = 8
    ):
        """
        Query documents with optional filtering.
        
        Args:
            query: Search query
            user_id: Filter by user (optional)
            document_id: Filter by specific document (optional)
            ticker: Filter by ticker symbol (optional)
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter
        filter_dict = {}
        if document_id:
            filter_dict["document_id"] = document_id
        elif user_id:
            filter_dict["user_id"] = user_id
        
        if ticker:
            filter_dict["ticker"] = ticker.upper()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Extract and format results
        chunks = []
        for match in results.matches:
            chunks.append({
                "content": match.metadata["text"],
                "score": match.score,
                "document_id": match.metadata["document_id"],
                "ticker": match.metadata["ticker"],
                "page": match.metadata.get("page", 0),
                "chunk_index": match.metadata["chunk_index"],
                "s3_key": match.metadata.get("s3_key", "")
            })
        
        return chunks
    
    def get_document_metadata(self, document_id: str, user_id: Optional[str] = None):
        """
        Get metadata for a specific document.
        
        Args:
            document_id: Document ID
            user_id: Optional user ID for ownership verification
            
        Returns:
            Document metadata or None if not found
        """
        filter_dict = {"document_id": document_id}
        if user_id:
            filter_dict["user_id"] = user_id
        
        # Query one vector to get metadata
        results = self.index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter=filter_dict
        )
        
        if results.matches:
            meta = results.matches[0].metadata
            return {
                "document_id": meta["document_id"],
                "ticker": meta["ticker"],
                "filename": meta["filename"],
                "s3_key": meta.get("s3_key", ""),
                "created_at": meta.get("created_at", "")
            }
        
        return None
    
    def delete_document(self, document_id: str, user_id: Optional[str] = None):
        """
        Delete all vectors and S3 file for a document.
        
        Args:
            document_id: Document to delete
            user_id: Optional user ID for ownership verification
        """
        # Get document metadata first (to get S3 key)
        doc_meta = self.get_document_metadata(document_id, user_id)
        
        if not doc_meta:
            logger.warning(f"Document {document_id} not found or access denied")
            return
        
        # Delete from S3 if exists
        s3_key = doc_meta.get("s3_key")
        if s3_key:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
                logger.info(f"Deleted from S3: {s3_key}")
            except ClientError as e:
                logger.error(f"Failed to delete from S3: {e}")
        
        # Delete vectors from Pinecone
        filter_dict = {"document_id": document_id}
        if user_id:
            filter_dict["user_id"] = user_id
        
        self.index.delete(filter=filter_dict)
        logger.info(f"✓ Deleted document: {document_id}")
    
    def list_user_documents(self, user_id: str):
        """
        List all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of document metadata
        """
        results = self.index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,
            include_metadata=True,
            filter={"user_id": user_id}
        )
        
        # Deduplicate by document_id
        seen = set()
        documents = []
        for match in results.matches:
            doc_id = match.metadata["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                documents.append({
                    "document_id": doc_id,
                    "ticker": match.metadata["ticker"],
                    "filename": match.metadata["filename"],
                    "created_at": match.metadata.get("created_at", ""),
                    "s3_key": match.metadata.get("s3_key", "")
                })
        
        # Sort by created_at (newest first)
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return documents
    
    def download_document(self, document_id: str, user_id: str, local_path: str):
        """
        Download a document's PDF from S3.
        
        Args:
            document_id: Document ID
            user_id: User ID (for ownership verification)
            local_path: Where to save the file locally
            
        Returns:
            Local file path or None if failed
        """
        doc_meta = self.get_document_metadata(document_id, user_id)
        
        if not doc_meta:
            logger.error(f"Document {document_id} not found or access denied")
            return None
        
        s3_key = doc_meta.get("s3_key")
        if not s3_key:
            logger.error(f"Document {document_id} has no S3 file")
            return None
        
        try:
            return self._download_from_s3(s3_key, local_path)
        except Exception as e:
            logger.error(f"Failed to download document: {e}")
            return None