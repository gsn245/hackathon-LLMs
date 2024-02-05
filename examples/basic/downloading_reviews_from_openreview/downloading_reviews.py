from pathlib import Path
from llms import load_openreview_credentials

import openreview

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.resolve()

# Loading the credentials
username, password = load_openreview_credentials()

# Creating a client for OpenReview
client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=username,
    password=password,
)

# Getting all submissions to the conference.
venue_id = "ICLR.cc/2024/Conference"
venue_group = client.get_group(venue_id)
submission_name = venue_group.content["submission_name"]["value"]

# Reference for the schema of submissions:
# https://docs.openreview.net/reference/api-v2/openapi-definition#notes
submissions = client.get_all_notes(
    invitation=f"{venue_id}/-/{submission_name}", details="replies"
)

review_name = venue_group.content["review_name"]["value"]

# Printing the reviews for the first 10 submissions.
for submission in submissions[:10]:
    print("-" * 50)
    print(submission.id)
    print(submission.content["title"]["value"])

    for reply in submission.details["replies"]:
        if (
            f"{venue_id}/{submission_name}{submission.number}/-/{review_name}"
            in reply["invitations"]
        ):
            # Printing the rating
            print(reply["content"]["rating"]["value"])

            # Printing the confidence
            # print(reply["content"]["confidence"]["value"])
