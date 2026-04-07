import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def append_audit_record(record: dict) -> str:
    log_id = str(uuid4())
    path = Path(os.getenv("MEDICAL_AI_AUDIT_LOG", "data/processed/prediction_audit.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": log_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return log_id