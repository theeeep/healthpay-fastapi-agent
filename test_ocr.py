import asyncio
import logging

from app.module.process_claim.services.ocr import process_ocr

# Set up logging
logging.basicConfig(level=logging.INFO)


async def test_ocr():
    """Test OCR extraction to see what text is being extracted."""
    # You can replace this with your actual PDF file path
    pdf_path = "test_bill.pdf"  # Replace with your actual PDF file

    try:
        with open(pdf_path, "rb") as f:
            content = f.read()

        ocr_text = await process_ocr(content, pdf_path)
        print("=== OCR EXTRACTED TEXT ===")
        print(ocr_text)
        print("=== END OCR TEXT ===")
        print(f"Text length: {len(ocr_text)} characters")

    except FileNotFoundError:
        print(f"File {pdf_path} not found. Please provide a valid PDF file path.")
    except Exception as e:
        print(f"Error processing OCR: {e}")


if __name__ == "__main__":
    asyncio.run(test_ocr())
