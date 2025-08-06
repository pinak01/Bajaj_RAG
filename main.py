from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import os
import tempfile
import requests
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
import concurrent.futures

# Import your custom modules
from Ragger import RAGRetriever
from miniCPM import Extract

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]  # Changed to simple list of strings
    processing_time: float

def get_openai_api_key():
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return api_key

def get_auth_token():
    """Get authentication token from environment variables"""
    auth_token = os.getenv("AUTH_TOKEN")
    if not auth_token:
        # Default token if not set in environment
        auth_token = "a2f387310984b739ae7e4accffad70a62e5673145dd05bc749dc913c0e6d0c42"
    return auth_token

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authentication token"""
    expected_token = get_auth_token()
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

async def download_document_async(document_url: str, temp_dir: str) -> str:
    """Asynchronously download document"""
    loop = asyncio.get_event_loop()
    
    def download():
        response = requests.get(str(document_url), timeout=30)  # Reduced timeout
        response.raise_for_status()
        
        filename = os.path.basename(str(document_url).split('?')[0])
        if not filename.endswith('.pdf'):
            filename = 'document.pdf'
        
        pdf_path = os.path.join(temp_dir, filename)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    
    return await loop.run_in_executor(None, download)

def process_single_question(rag_retriever, question: str, hint: str = "") -> str:
    """Process a single question - for parallel execution"""
    try:
        return rag_retriever.mRetriever(Q=question, hint=hint)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Error: {str(e)}"

def process_document_and_questions(
    document_url: str,
    questions: List[str]
) -> List[str]:
    """Process document and execute multiple queries with optimizations"""
    openai_api_key = get_openai_api_key()
    
    # Optimized configuration
    chunk_size = 500  # Reduced chunk size
    chunk_overlap = 100  # Reduced overlap
    method = "text"
    generator_model = "gpt-3.5-turbo"  # Faster model
    temperature = 0
    hint = ""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download document
        try:
            logger.info(f"Downloading document from: {document_url}")
            response = requests.get(str(document_url), timeout=30)  # Reduced timeout
            response.raise_for_status()
            
            filename = os.path.basename(str(document_url).split('?')[0])
            if not filename.endswith('.pdf'):
                filename = 'document.pdf'
            
            pdf_path = os.path.join(temp_dir, filename)
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Document downloaded: {len(response.content)} bytes")
            
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        
        # Extract text with smart fallback
        try:
            logger.info("Extracting text from document...")
            extractor = Extract(pdf_path, method='text', openai_api_key=openai_api_key)
            
            # Use smart extraction with automatic fallback
            doc_text = extractor.extract_with_fallback()
            
            logger.info(f"Text extracted: {len(doc_text)} characters")
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
        
        # Save text to temporary file
        text_path = os.path.join(temp_dir, "document.txt")
        with open(text_path, "w", encoding='utf-8') as text_file:
            text_file.write(doc_text)
        
        # Create RAG retriever with optimizations
        try:
            logger.info("Creating RAG retriever...")
            rag_retriever = RAGRetriever(
                text_path,
                device='cuda',
                embed_device='cuda',
                vecStoreDir=temp_dir,
                GeneratorModel=generator_model,  # Using faster model
                temp=temperature,
                openai_api_key=openai_api_key
            )
            
            # Create vector store with smaller chunks
            rag_retriever.createVecStore_multiVec(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            logger.info("RAG retriever created")
            
        except Exception as e:
            logger.error(f"Error creating RAG retriever: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create RAG retriever: {str(e)}")
        
        # Process questions in parallel
        results = []
        try:
            logger.info(f"Processing {len(questions)} questions in parallel...")
            
            # Use ThreadPoolExecutor for parallel question processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(questions), 4)) as executor:
                # Submit all questions for parallel processing
                future_to_question = {
                    executor.submit(process_single_question, rag_retriever, question, hint): question 
                    for question in questions
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_question, timeout=25):
                    try:
                        answer = future.result()
                        results.append(answer)
                    except Exception as e:
                        logger.error(f"Question processing failed: {e}")
                        results.append(f"Error: {str(e)}")
            
            logger.info("All questions processed")
            return results
            
        except Exception as e:
            logger.error(f"Error executing queries: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to execute queries: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Optimized RAG Document Query API",
    description="Fast RAG API for document querying",
    version="2.0.0"
)

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Query a document using optimized RAG with authentication
    
    Returns a simple array of answers for faster processing.
    """
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Process document and execute queries
        answers = await asyncio.get_event_loop().run_in_executor(
            None,
            process_document_and_questions,
            str(request.documents),
            request.questions
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed in {processing_time:.2f} seconds")
        
        return QueryResponse(
            answers=answers,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_run: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint (no authentication required)"""
    return {"status": "healthy", "service": "Optimized RAG Document Query API"}

if __name__ == "__main__":
    import uvicorn
    
    # Check for OpenAI API key on startup
    try:
        get_openai_api_key()
        logger.info("OpenAI API key found")
    except ValueError as e:
        logger.error(f"Startup failed: {e}")
        exit(1)
    
    # Log the authentication token being used
    auth_token = get_auth_token()
    logger.info(f"Using auth token: {auth_token[:10]}...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )