import fitz
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_pdf_text(uploaded_file):
    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        page_text = page.get_text()

        # check if extracted text is readable
        if page_text.strip() and not page_text.startswith(""):
            text += page_text

        else:
            # OCR fallback for scanned pdfs
            pix = page.get_pixmap(
                matrix=fitz.matrix(2, 2)
            )    

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text

    pdf.close()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks