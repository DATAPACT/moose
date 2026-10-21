import os
import time

import httpx


def main():
    api_key = os.environ.get("MOOSE_API_KEY")
    if not api_key:
        raise RuntimeError("MOOSE_API_KEY is required")
    
    payload = {
        "schema": "coarse",
        "text": "Roberto Avogadro founded Moose in 2024.",
    }
    
    headers = {"X-API-Key": api_key}
    
    with httpx.Client() as client:
        resp = client.post("http://localhost:8000/ner", json=payload, headers=headers)
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
