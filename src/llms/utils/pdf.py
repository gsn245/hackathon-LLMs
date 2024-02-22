"""Utilities for loading pdfs as strings."""

from typing import Iterable

from PyPDF2 import PdfReader


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb
def read_pdf(
    filepath: str, pages: Iterable[int] = None, page_separator: str = "\n"
) -> str:
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    for page_number, page in enumerate(reader.pages):
        if pages is not None:
            if (page_number + 1) not in pages:
                continue

        pdf_text += page.extract_text() + page_separator
    return pdf_text
