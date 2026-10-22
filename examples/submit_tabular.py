import os
import time

import httpx


def main():
    api_key = os.environ.get("MOOSE_API_KEY")
    if not api_key:
        raise RuntimeError("MOOSE_API_KEY is required")
    
    payload = {
        "schema": "coarse",
        "table_id": "employees",
        "sampled_rows": [
            {"name": "Alice Smith", "email": "alice@example.com", "age": "29"},
            {"name": "Bob Jones", "email": "bob@example.com", "age": "41"},
        ],
    }
    
    headers = {"X-API-Key": api_key}
    
    with httpx.Client() as client:
        resp = client.post(
            "http://localhost:8000/tabular/annotate", json=payload, headers=headers
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print("job_id:", job_id)

        while True:
            status = client.get(f"http://localhost:8000/jobs/{job_id}", headers=headers)
            status.raise_for_status()
            data = status.json()
            print("status:", data["status"])
            if data["status"] in {"completed", "failed"}:
                print(data)
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
