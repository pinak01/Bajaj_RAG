import torch
from PIL import Image
from openai import OpenAI
from pdf2image import convert_from_path
import pathlib
from PyPDF2 import PdfReader
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class Extract():
    def __init__(self, pdf_path, device='cuda', method='text', openai_api_key=None):
        if method == 'ocr':
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OCR method")
            self.client = OpenAI(api_key=openai_api_key)
        self.path = pdf_path
        self.method = method
        
    def getText(self):
        """Optimized text extraction with better error handling"""
        try:
            reader = PdfReader(self.path)
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
                    text_parts.append(f'\n--- Page {i+1} ---\n')
            
            return ''.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_single_image(self, page_image, page_num):
        """Process a single page image with OCR"""
        try:
            # Convert PIL image to base64 more efficiently
            img_byte_arr = io.BytesIO()
            
            # Optimize image before sending to API
            # Resize if too large (reduces API processing time and cost)
            max_size = 1024
            if max(page_image.size) > max_size:
                page_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Use JPEG instead of PNG for better compression (faster upload)
            page_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Simplified, more efficient prompt
            question = "Extract all text from this image. Return only the text content."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Faster and cheaper than gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,  # Reduced from 4096
                temperature=0.1   # Reduced for more consistent output
            )
            
            res = response.choices[0].message.content
            return f"\n--- Page {page_num} ---\n{res}\n"
            
        except Exception as e:
            logger.error(f"Error processing page {page_num} with OCR: {e}")
            return f"\n--- Page {page_num} (Error) ---\nError processing page with OCR\n"
    
    def getTextFromImg(self):
        """Optimized OCR with parallel processing and image optimization"""
        try:
            # Convert PDF to images with optimized settings
            if pathlib.Path(self.path).suffix == '.pdf':
                # Reduced DPI from 800 to 300 for faster processing while maintaining quality
                images = convert_from_path(
                    self.path, 
                    dpi=300,  # Reduced from 800
                    first_page=None,
                    last_page=None,
                    fmt='JPEG',  # Use JPEG format for memory efficiency
                    thread_count=2  # Limit thread count to prevent memory issues
                )
            else:
                images = [Image.open(self.path).convert('RGB')]
            
            logger.info(f"Processing {len(images)} pages with OCR")
            
            # Process images in parallel for faster OCR
            # Limit workers to prevent API rate limiting and memory issues
            max_workers = min(3, len(images))  # Max 3 concurrent API calls
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pages for processing
                future_to_page = {
                    executor.submit(self.process_single_image, image, i+1): i 
                    for i, image in enumerate(images)
                }
                
                # Collect results in order
                page_results = [None] * len(images)
                for future in future_to_page:
                    page_num = future_to_page[future]
                    try:
                        page_results[page_num] = future.result(timeout=30)  # 30 sec timeout per page
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        page_results[page_num] = f"\n--- Page {page_num + 1} (Error) ---\nTimeout or error\n"
            
            # Combine all results
            document_text = ''.join(filter(None, page_results))
            logger.info(f"OCR completed. Extracted {len(document_text)} characters")
            
            return document_text
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return f"Error during OCR processing: {str(e)}"
        
    def extract_with_fallback(self):
        """Smart extraction with automatic fallback"""
        # Try text extraction first (much faster)
        text = self.getText()
        
        # Check if text extraction was successful
        if len(text.strip()) > 100:  # Reasonable amount of text found
            logger.info("Text extraction successful")
            return text
        else:
            logger.info("Text extraction yielded poor results, using OCR fallback")
            return self.getTextFromImg()
    
    def quick_text_check(self):
        """Quick check to see if PDF has extractable text"""
        try:
            reader = PdfReader(self.path)
            # Check first few pages for text content
            pages_to_check = min(3, len(reader.pages))
            total_chars = 0
            
            for i in range(pages_to_check):
                page_text = reader.pages[i].extract_text()
                total_chars += len(page_text.strip())
                
                # If we find reasonable amount of text, assume PDF has text
                if total_chars > 200:
                    return True
                    
            return total_chars > 50  # Minimum threshold
            
        except Exception:
            return False