from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import os
import tempfile
import requests
import asyncio
import logging
import hashlib
import shutil
from datetime import datetime
from dotenv import load_dotenv
import concurrent.futures
from urllib.parse import urlparse
import json

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

# Global cache directory
CACHE_DIR = os.path.join(os.getcwd(), "document_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache for RAG retrievers to avoid recreating them
rag_cache: Dict[str, RAGRetriever] = {}

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

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

def get_document_hash(document_url: str) -> str:
    """
    Create a unique hash for a document URL, ignoring query parameters
    that might change (like timestamps) but keeping the core document identity
    """
    parsed_url = urlparse(document_url)
    # Use the path part of the URL to identify the document
    base_path = parsed_url.path
    
    # Create a hash from the base path
    return hashlib.md5(base_path.encode()).hexdigest()

def get_document_name_from_url(document_url: str) -> str:
    """Extract a human-readable document name from URL"""
    parsed_url = urlparse(document_url)
    filename = os.path.basename(parsed_url.path)
    
    # Remove .pdf extension and decode URL encoding
    if filename.endswith('.pdf'):
        filename = filename[:-4]
    
    # Handle URL encoding
    import urllib.parse
    filename = urllib.parse.unquote(filename)
    
    # Clean up the filename
    filename = filename.replace('%20', '_').replace(' ', '_')
    
    return filename or 'document'

def get_cache_paths(document_hash: str):
    """Get cache file paths for a document"""
    cache_subdir = os.path.join(CACHE_DIR, document_hash)
    return {
        'cache_dir': cache_subdir,
        'pdf_path': os.path.join(cache_subdir, 'document.pdf'),
        'text_path': os.path.join(cache_subdir, 'document.txt'),
        'vector_store_dir': os.path.join(cache_subdir, 'vector_store'),
        'metadata_path': os.path.join(cache_subdir, 'metadata.json')
    }

def is_cache_valid(metadata_path: str, max_age_hours: int = 24) -> bool:
    """Check if cached document is still valid"""
    if not os.path.exists(metadata_path):
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        cached_time = datetime.fromisoformat(metadata['cached_at'])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        return age_hours < max_age_hours
    except:
        return False

def cache_document(document_url: str, document_hash: str) -> Dict[str, str]:
    """Download and cache a document"""
    cache_paths = get_cache_paths(document_hash)
    
    # Create cache directory
    os.makedirs(cache_paths['cache_dir'], exist_ok=True)
    
    try:
        # Download document
        logger.info(f"Downloading and caching document: {document_url}")
        response = requests.get(str(document_url), timeout=30)
        response.raise_for_status()
        
        # Save PDF
        with open(cache_paths['pdf_path'], 'wb') as f:
            f.write(response.content)
        
        # Extract text
        logger.info("Extracting text from cached document...")
        openai_api_key = get_openai_api_key()
        extractor = Extract(cache_paths['pdf_path'], method='text', openai_api_key=openai_api_key)
        doc_text = extractor.extract_with_fallback()
        
        # Save text
        with open(cache_paths['text_path'], 'w', encoding='utf-8') as f:
            f.write(doc_text)
        
        # Save metadata
        metadata = {
            'document_url': document_url,
            'document_name': get_document_name_from_url(document_url),
            'cached_at': datetime.now().isoformat(),
            'text_length': len(doc_text),
            'pdf_size': len(response.content)
        }
        
        with open(cache_paths['metadata_path'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Document cached successfully: {len(doc_text)} characters")
        return cache_paths
        
    except Exception as e:
        # Clean up partial cache on error
        if os.path.exists(cache_paths['cache_dir']):
            shutil.rmtree(cache_paths['cache_dir'])
        raise e

def get_or_create_rag_retriever(document_hash: str, cache_paths: Dict[str, str]) -> RAGRetriever:
    """Get RAG retriever from memory cache or create new one - optimized for speed"""
    
    # Check in-memory cache first
    if document_hash in rag_cache:
        logger.info(f"Using in-memory RAG retriever for document {document_hash}")
        return rag_cache[document_hash]
    
    # Create new RAG retriever with speed-optimized settings
    logger.info(f"Creating new RAG retriever for document {document_hash}")
    openai_api_key = get_openai_api_key()
    
    # Ultra-speed optimized configuration
    chunk_size = 800  # Larger chunks = fewer chunks = faster
    chunk_overlap = 30  # Minimal overlap for maximum speed
    generator_model = "gpt-3.5-turbo"  # Fastest model
    temperature = 0  # No randomness for speed
    
    rag_retriever = RAGRetriever(
        cache_paths['text_path'],
        device='cuda',
        embed_device='cuda',
        vecStoreDir=cache_paths['vector_store_dir'],
        GeneratorModel=generator_model,
        temp=temperature,
        openai_api_key=openai_api_key
    )
    
    # Check if vector store exists
    if os.path.exists(cache_paths['vector_store_dir']) and os.listdir(cache_paths['vector_store_dir']):
        logger.info("Loading existing vector store from cache")
        # The RAGRetriever should automatically load existing vector store
    else:
        logger.info("Creating new vector store with speed optimization")
        rag_retriever.createVecStore(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    # Store in memory cache (with larger size limit for speed)
    if len(rag_cache) >= 10:  # Increased from 5 to 10
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(rag_cache))
        del rag_cache[oldest_key]
        logger.info(f"Removed oldest RAG retriever from memory cache: {oldest_key}")
    
    rag_cache[document_hash] = rag_retriever
    logger.info(f"Added RAG retriever to memory cache: {document_hash}")
    
    return rag_retriever

def process_single_question(rag_retriever, question: str, hint: str = "") -> str:
    """Process a single question - for parallel execution"""
    try:
        return rag_retriever.mRetriever(Q=question, hint=hint)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Error: {str(e)}"

def process_single_question_fast(args):
    """Optimized single question processor for multiprocessing"""
    rag_retriever, question, hint, index = args
    try:
        answer = rag_retriever.mRetriever(Q=question, hint=hint)
        return index, answer
    except Exception as e:
        return index, f"Error: {str(e)}"

def process_questions_ultra_fast(rag_retriever, questions: List[str], hint: str = "", max_time: int = 25) -> List[str]:
    """Ultra-fast question processing with aggressive parallelization and timeout"""
    logger.info(f"Processing {len(questions)} questions with {max_time}s timeout")
    
    # Calculate optimal worker count - more aggressive for speed
    optimal_workers = min(len(questions), 8)  # Increased from 2 to 8
    
    results = [f"Error: Timeout processing question {i+1}" for i in range(len(questions))]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all tasks immediately
            future_to_index = {
                executor.submit(process_single_question, rag_retriever, question, hint): i 
                for i, question in enumerate(questions)
            }
            
            completed_count = 0
            start_time = datetime.now()
            
            # Process with strict timeout
            for future in concurrent.futures.as_completed(future_to_index, timeout=max_time):
                try:
                    # Individual question timeout - very short
                    answer = future.result(timeout=3)  # 3 seconds per question max
                    index = future_to_index[future]
                    results[index] = answer
                    completed_count += 1
                    
                    # Check if we're running out of time
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > max_time * 0.9:  # Stop at 90% of max time
                        logger.warning(f"Approaching time limit at {elapsed:.1f}s, stopping new processing")
                        break
                        
                except Exception as e:
                    index = future_to_index[future]
                    results[index] = f"Error: {str(e)}"
                    completed_count += 1
            
            # Cancel any remaining futures
            for future in future_to_index:
                if not future.done():
                    future.cancel()
                    
            logger.info(f"Ultra-fast processing completed: {completed_count}/{len(questions)} in {(datetime.now() - start_time).total_seconds():.2f}s")
                    
    except concurrent.futures.TimeoutError:
        logger.warning(f"Ultra-fast processing timeout after {max_time}s")
    except Exception as e:
        logger.error(f"Ultra-fast processing error: {e}")
    
    return results

def process_document_and_questions_cached(
    document_url: str,
    questions: List[str]
) -> List[str]:  # Returns only answers
    """Process document and execute multiple queries with ultra-fast caching"""
    
    # Get document hash for caching
    document_hash = get_document_hash(document_url)
    cache_paths = get_cache_paths(document_hash)
    
    cache_hit = False
    
    # Check if document is cached and valid
    if (os.path.exists(cache_paths['text_path']) and 
        os.path.exists(cache_paths['pdf_path']) and
        is_cache_valid(cache_paths['metadata_path'])):
        
        logger.info(f"Using cached document: {document_hash}")
        cache_hit = True
    else:
        logger.info(f"Cache miss for document: {document_hash}")
        cache_paths = cache_document(document_url, document_hash)
    
    # Get or create RAG retriever
    try:
        start_retriever_time = datetime.now()
        rag_retriever = get_or_create_rag_retriever(document_hash, cache_paths)
        retriever_time = (datetime.now() - start_retriever_time).total_seconds()
        logger.info(f"RAG retriever ready in {retriever_time:.2f}s")
    except Exception as e:
        logger.error(f"Error creating RAG retriever: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create RAG retriever: {str(e)}")
    
    # Ultra-fast question processing with strict time limit
    try:
        # Reserve time for setup, give remaining time to questions
        max_question_time = max(5, 25 - int(retriever_time))  # At least 5s, max 25s
        logger.info(f"Processing {len(questions)} questions with {max_question_time}s limit")
        
        results = process_questions_ultra_fast(rag_retriever, questions, "", max_question_time)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in ultra-fast processing: {e}")
        # Emergency fallback - return error messages quickly
        fallback_results = [f"Error: System overload - {str(e)}" for _ in range(len(questions))]
        return fallback_results

# Initialize FastAPI app
app = FastAPI(
    title="Cached RAG Document Query API",
    description="Fast RAG API with document caching for improved performance",
    version="3.0.0"
)

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Query a document using ultra-fast cached RAG with strict 30s timeout"""
    
    # Set absolute timeout of 29 seconds (leaving 1s buffer)
    timeout_seconds = 29
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing request with {len(request.questions)} questions (timeout: {timeout_seconds}s)")
        
        # Use asyncio timeout to enforce strict time limit
        try:
            answers = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    process_document_and_questions_cached,
                    str(request.documents),
                    request.questions
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {timeout_seconds} seconds")
            # Return partial/error responses for timeout
            answers = [f"Error: Request timeout - question {i+1} not processed" 
                      for i in range(len(request.questions))]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed in {processing_time:.2f} seconds")
        
        return QueryResponse(
            answers=answers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Unexpected error in hackrx_run after {processing_time:.2f}s: {e}")
        
        # Return error responses quickly
        error_answers = [f"Error: System error - {str(e)}" for _ in range(len(request.questions))]
        return QueryResponse(
            answers=error_answers
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Cached RAG Document Query API"}

@app.get("/cache/status")
async def cache_status(token: str = Depends(verify_token)):
    """Get cache status information"""
    cache_info = []
    
    if os.path.exists(CACHE_DIR):
        for item in os.listdir(CACHE_DIR):
            item_path = os.path.join(CACHE_DIR, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        cache_info.append({
                            'document_hash': item,
                            'document_name': metadata.get('document_name', 'Unknown'),
                            'cached_at': metadata.get('cached_at'),
                            'is_valid': is_cache_valid(metadata_path),
                            'text_length': metadata.get('text_length', 0),
                            'in_memory': item in rag_cache
                        })
                    except:
                        pass
    
    return {
        'cache_directory': CACHE_DIR,
        'cached_documents': len(cache_info),
        'in_memory_retrievers': len(rag_cache),
        'documents': cache_info
    }

@app.delete("/cache/clear")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear all cached documents"""
    try:
        # Clear in-memory cache
        rag_cache.clear()
        
        # Clear disk cache
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        logger.info("Cache cleared successfully")
        return {"message": "Cache cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

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
    logger.info(f"Cache directory: {CACHE_DIR}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )