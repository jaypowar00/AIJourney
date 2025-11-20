import csv
import os
import uuid
from typing import List, Dict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "docs.csv")

os.makedirs(DATA_DIR, exist_ok=True)

FIELDNAMES = ["id", "content", "created_at"]


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def save_doc(content: str) -> Dict:
    """Save a new document and return its record."""
    _ensure_csv()
    doc_id = str(uuid.uuid4())
    rec = {"id": doc_id, "content": content, "created_at": datetime.utcnow().isoformat()}
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(rec)
    return rec


def load_documents() -> List[Dict]:
    """Load all docs from CSV as list of records (id, content, created_at)."""
    _ensure_csv()
    out = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({"id": r["id"], "content": r["content"], "created_at": r.get("created_at")})
    return out


if __name__ == "__main__":
    # quick local smoke test
    save_doc("This is a test document")
    print(load_documents())
