"""Where the raw data comes from and which files the project uses."""

KAGGLE_DATASET = "wordsforthewise/lending-club"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
LICENSE = "CC0: Public Domain"

# Accepted loans, 2007 to 2018Q4, one row per loan, status as of the 2018Q4 snapshot.
# Kept gzip-compressed: pandas reads it directly and it is ~5x smaller on disk.
ACCEPTED_FILE = "accepted_2007_to_2018Q4.csv.gz"

# The archive also holds rejected applications; they have no outcome, so they are not used.
SOURCE_FILES = (ACCEPTED_FILE,)
