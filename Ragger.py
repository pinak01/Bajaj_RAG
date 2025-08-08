from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryByteStore
from langchain_core.prompts import ChatPromptTemplate
import uuid
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

class RAGRetriever():
    def __init__(self, text_path, device='cuda', embed_device='cpu', vecStoreDir='./', 
                 GeneratorModel='gpt-3.5-turbo', temp=0, openai_api_key=None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        # Load text with UTF-8 encoding
        with open(text_path, "r", encoding='utf-8') as f:
            self.text_txt = f.read()
        
        # Load documents
        self.text = TextLoader(text_path, encoding='utf-8').load()
        
        # Use smallest, fastest embedding model
        self.embedding_function = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small",
            dimensions=256  # Reduced dimensions for maximum speed
        )

        self.vecStoreDir = vecStoreDir
        # Optimized for speed
        self.chatModel = ChatOpenAI(
            model=GeneratorModel, 
            temperature=temp,
            openai_api_key=openai_api_key,
            max_tokens=200,  # Balanced for speed vs quality
            request_timeout=10  # Add timeout for faster failures
        )
    
    def update_chat_model(self, GeneratorModel='gpt-3.5-turbo', temp=0, device='cuda', openai_api_key=None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.chatModel = ChatOpenAI(
            model=GeneratorModel, 
            temperature=temp, 
            openai_api_key=openai_api_key,
            max_tokens=200,
            request_timeout=10
        )

    def get_chunks(self, chunk_size=600, chunk_overlap=50):
        """Speed-optimized chunking with larger chunks, less overlap"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(self.text)
        self.chunks = texts
        return texts
    
    def createVecStore_multiVec(self, chunk_size=600, chunk_overlap=50):
        """Speed-optimized vector store - use simple retrieval for better performance"""
        # For speed, use simple vector store instead of multi-vector
        return self.createVecStore(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def createVecStore(self, chunk_size=600, chunk_overlap=50):
        """Speed-optimized simple vector store"""
        chunks = self.get_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        logging.info(f"Creating vector store with {len(chunks)} chunks")
        
        # For speed, process in smaller batches with faster settings
        if len(chunks) > 3000:  # Reduced threshold
            logging.info(f"Large document detected ({len(chunks)} chunks). Using batch processing.")
            
            # Create empty Chroma DB first
            self.db = Chroma(
                collection_name="speed_retrieval",
                embedding_function=self.embedding_function
            )
            
            # Smaller batches for faster processing
            batch_size = 2000  # Reduced batch size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                self.db.add_documents(batch)
        else:
            # For smaller documents, use the original method
            self.db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                collection_name="speed_retrieval"
            )
        
        # Create simple retriever for speed
        self.MultiRetriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Balanced retrieval
        )
            
        return self.db
    
    def mRetriever(self, Q='query', hint=''):
        """Speed-optimized retriever with efficient prompt"""
        RETRIEVER = self.MultiRetriever 
        
        # Speed-optimized prompt - shorter and more direct
        template = """Answer the question based on the context provided. Be specific and informative but concise.

Context: {context}

Question: {question}

Answer:"""
        
        if len(hint) > 0:
            template = template + f'\nHint: {hint}'
            
        prompt = ChatPromptTemplate.from_template(template)
        self.prompt = prompt
        self.template = template
        
        def format_docs(docs):
            # Optimized context processing for speed
            context = "\n\n".join(doc.page_content for doc in docs)
            # Reasonable context limit for speed vs quality balance
            return context[:2500] if len(context) > 2500 else context

        chain = {
            "context": RETRIEVER | format_docs,
            "question": RunnablePassthrough()
        } | prompt | self.chatModel | StrOutputParser()
        
        return chain.invoke(Q)

