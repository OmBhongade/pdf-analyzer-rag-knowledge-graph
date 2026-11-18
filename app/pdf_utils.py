from pypdf import PdfReader

def extract_pdf_text(file_path: str) -> list[dict]:
    """Return one record per non-empty PDF page."""
    reader = PdfReader(file_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "source": file_path.split("/")[-1]
            })
    return pages
