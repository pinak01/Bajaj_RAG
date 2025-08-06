# SimpleRAG with OpenAI Models

This is a simple RAG (Retrieval-Augmented Generation) system that uses OpenAI models for text extraction, embeddings, and generation.

## Features

- **PDF Text Extraction**: Extract text from PDFs using either direct text extraction or OCR with GPT-4 Vision
- **Vector Storage**: Create and manage vector embeddings using OpenAI's text-embedding-3-small
- **RAG Retrieval**: Multi-vector retrieval system with semantic search
- **OpenAI Integration**: Uses GPT-4 for text generation and GPT-4 Vision for OCR

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:

   - Create a `.env` file in the project root directory
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your-actual-openai-api-key-here
     ```
   - Make sure to replace `your-actual-openai-api-key-here` with your real OpenAI API key

3. **Place Documents**:
   - Put your PDF documents in the `documents/` folder

## Usage

1. **Run the main script**:

   ```bash
   python main.py
   ```

2. **Customize the query**:
   - Edit the `query` variable in `main.py` to ask your specific question
   - Modify the `hint` variable to provide additional instructions to the model

## Configuration

### Models Used:

- **Text Generation**: GPT-4 (configurable in main.py)
- **OCR**: GPT-4 Vision (for image-based text extraction)
- **Embeddings**: OpenAI text-embedding-3-small

### Parameters:

- `chunk_size`: Length of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 500 characters)
- `temp`: Temperature for text generation (default: 0)

## File Structure

- `main.py`: Main execution script
- `miniCPM.py`: Text extraction module (now uses OpenAI)
- `Ragger.py`: RAG retrieval system (now uses OpenAI)
- `requirements.txt`: Python dependencies
- `.env`: Environment variables file (create this file with your API key)
- `documents/`: Folder for PDF documents

## Notes

- The system automatically falls back to OCR if text extraction yields insufficient content
- Vector store is created once and reused for subsequent queries
- All models are now cloud-based through OpenAI API
- Make sure you have sufficient OpenAI API credits for your usage
- The `.env` file should be added to `.gitignore` to keep your API key secure
