"""An example of how to load a pdf as a python string"""
from pathlib import Path

from llms.utils.pdf import read_pdf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.resolve()

# Load the pdf as a string
score_based = read_pdf(ROOT_DIR / "examples" / "reading_a_pdf" / "score_based.pdf")

print(score_based[:1000])
