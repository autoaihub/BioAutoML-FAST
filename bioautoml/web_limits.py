"""Per-set upload limit used by the web submission forms and help."""
import csv
import os

DEFAULT_SAMPLE_LIMIT = 5_000
# Only an explicit opt-out disables the default.
SAMPLE_LIMIT = None if os.environ.get("BIOAUTOML_LIMIT_SAMPLES") == "no" else DEFAULT_SAMPLE_LIMIT

SAMPLE_LIMIT_TEXT = (
    f"Each training, test, or prediction set may contain up to {SAMPLE_LIMIT:,} samples "
    "(FASTA sequences or CSV data rows), counted across its files."
    if SAMPLE_LIMIT else "This deployment has no sample-count limit per set."
)
PREDICTION_LIMIT_TEXT = (
    f"Each prediction set may contain up to {SAMPLE_LIMIT:,} sequences."
    if SAMPLE_LIMIT else "This deployment has no sequence-count limit per prediction set."
)


def sample_limit_error(files, data_type, set_name):
    if SAMPLE_LIMIT is None or not files:
        return None
    count = 0
    for stream in files if isinstance(files, list) else [files]:
        stream.seek(0)
        try:
            if data_type == "Structured data":
                rows = csv.reader(line.decode("utf-8") for line in stream)
                next(rows, None)  # Header
                count += sum(1 for row in rows if row)
            else:
                count += sum(line.startswith(b">") for line in stream)
        except (csv.Error, UnicodeDecodeError) as error:
            return f"{set_name} contains an unreadable CSV file: {error}"
        finally:
            stream.seek(0)
    if count > SAMPLE_LIMIT:
        return f"{set_name} exceeds the sample limit ({count:,} uploaded, limit {SAMPLE_LIMIT:,})."
    return None
