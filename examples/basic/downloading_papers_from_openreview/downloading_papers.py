from pathlib import Path
from llms import load_openreview_credentials

import openreview

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.resolve()

username, password = load_openreview_credentials()

# API V2
client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=username,
    password=password,
)

# Get the venue id from the URL of the conference
notes = client.get_all_notes(
    content={"venueid": "ICLR.cc/2024/Conference"}, details="original"
)

# Print the title of each paper
for note in notes:
    print(note.id, note.content["title"])

# Downloading the first 10 papers
SAVE_PATH = ROOT_DIR / "data" / "raw" / "downloaded_from_openreview"
SAVE_PATH.mkdir(parents=True, exist_ok=True)
for note in notes[:10]:
    pdf = client.get_pdf(note.id)
    with open(SAVE_PATH / f"{note.id}.pdf", "wb") as f:
        f.write(pdf)
