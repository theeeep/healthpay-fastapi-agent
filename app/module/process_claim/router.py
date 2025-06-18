from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.module.process_claim.services.ocr import process_ocr

process_claim_router = APIRouter()


@process_claim_router.post("/process-claim")
async def process_claim_documents(files: List[UploadFile] = File(...)):
    extracted_texts = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type: {file.filename}. Only PDF files are allowed.")

        if file.filename is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is missing a filename.")

        file_content = await file.read()

        # 1. OCR Step
        ocr_text = await process_ocr(file_content, file.filename)
        extracted_texts.append({"filename": file.filename, "extracted_text": ocr_text})

    return {"message": "OCR processing complete", "extracted_documents": extracted_texts}
