import base64

from mistralai import Mistral

from app.config.settings import Config
from app.core.logger import logger


async def process_ocr(file_content: bytes, filename: str) -> str:
    """Processes a PDF file using Mistral OCR.

    This function sends the base64-encoded PDF content to Mistral OCR
    and extracts text from it.

    Args:
        file_content: The content of the PDF file as bytes.
        filename: The name of the file.

    Returns:
        A string representing the extracted text.
    """
    try:
        mistral_api_key = Config.MISTRAL_API_KEY
        mistral_ocr_model = "mistral-ocr-latest"  # As per Mistral AI documentation

        async with Mistral(api_key=mistral_api_key) as mistral_client:
            # Encode PDF bytes as base64
            base64_pdf = base64.b64encode(file_content).decode("utf-8")

            logger.info(f"Processing PDF with Mistral OCR: {filename}")
            ocr_response = await mistral_client.ocr.process_async(
                model=mistral_ocr_model, document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"}
            )

            combined_text = ""
            if hasattr(ocr_response, "pages") and ocr_response.pages:
                for page in ocr_response.pages:
                    page_num = page.index
                    page_text = page.markdown
                    if page_text:
                        combined_text += f"[Page {page_num}]\n{page_text}\n\n"

            if not combined_text:
                logger.warning(f"Mistral OCR returned no text for PDF: {filename}")
                return ""

            logger.info(f"Successfully extracted text from PDF using Mistral OCR for {filename}")
            return combined_text

    except Exception as e:
        logger.error(f"Error in PDF OCR processing with Mistral for {filename}: {str(e)}")
        return ""
