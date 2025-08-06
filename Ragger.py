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
        
        # Use smaller, faster embedding model
        self.embedding_function = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small",
            dimensions=512  # Reduced dimensions for speed
        )

        self.vecStoreDir = vecStoreDir
        # Use faster model by default
        self.chatModel = ChatOpenAI(
            model=GeneratorModel, 
            temperature=temp,
            openai_api_key=openai_api_key,
            max_tokens=150  # Limit response length for speed
        )
    
    def update_chat_model(self, GeneratorModel='gpt-3.5-turbo', temp=0, device='cuda', openai_api_key=None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.chatModel = ChatOpenAI(
            model=GeneratorModel, 
            temperature=temp, 
            openai_api_key=openai_api_key,
            max_tokens=250  # Consistent token limit
        )

    def get_chunks(self, chunk_size=500, chunk_overlap=100):
        """Optimized chunking with smaller sizes"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(self.text)
        self.chunks = texts
        return texts
    
    def createVecStore_multiVec(self, chunk_size=500, chunk_overlap=100):
        """Optimized vector store creation"""
        store = InMemoryByteStore()
        id_key = "doc_id"
        
        # Get chunks with optimized parameters
        docs = self.get_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create retriever with optimized search parameters
        retriever = MultiVectorRetriever(
            vectorstore=Chroma(
                collection_name="summaries", 
                embedding_function=self.embedding_function
            ),
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": 2},  # Reduced from 4 to 2 for speed
            search_type='similarity'  # Changed from 'mmr' to 'similarity' for speed
        )

        doc_ids = [str(uuid.uuid4()) for _ in docs]
        
        # Reduced sub-chunking for speed
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,  # Reduced from 200
            chunk_overlap=25,  # Reduced from 50
            length_function=len,
            is_separator_regex=False
        )
        
        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id
            sub_docs.extend(_sub_docs)
        
        # Batch add documents for better performance
        retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        
        self.MultiRetriever = retriever

    def createVecStore(self, chunk_size=500, chunk_overlap=100):
        """Simplified vector store for basic retrieval"""
        self.db = Chroma.from_documents(
            self.get_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            self.embedding_function
        )
        return self.db
    
    def mRetriever(self, Q='query', hint=''):
        """Optimized retriever with simplified prompt"""
        RETRIEVER = self.MultiRetriever 
        
        # Simplified, faster prompt
        template = """Based on the context below, provide a concise answer to the question.
        
        Context: {context}
        Question: {question}
        
        Answer:"""
        
        if len(hint) > 0:
            template = template + f'\nHint: {hint}'
            
        prompt = ChatPromptTemplate.from_template(template)
        self.prompt = prompt
        self.template = template
        
        def format_docs(docs):
            # Limit context length for faster processing
            context = "\n\n".join(doc.page_content for doc in docs)
            # Truncate if too long (keep first 2000 chars)
            return context[:2000] if len(context) > 2000 else context

        chain = {
            "context": RETRIEVER | format_docs,
            "question": RunnablePassthrough()
        } | prompt | self.chatModel | StrOutputParser()
        
        return chain.invoke(Q)