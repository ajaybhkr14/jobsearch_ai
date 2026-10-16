import fitz  
from fastapi import HTTPException, UploadFile

async def extract_text_from_pdf(file: UploadFile) -> str:
    """Extracts raw text from an uploaded PDF file stream."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        extracted_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text("text") + "\n"
            
        pdf_document.close()
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF. Ensure it is not a scanned image without an OCR layer."
            )
            
        return extracted_text.strip()
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error parsing PDF file: {str(e)}")