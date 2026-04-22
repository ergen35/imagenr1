import base64
import binascii
import json
import logging
import os
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

app = FastAPI(title="Image Prompt App")
logger = logging.getLogger("app.gpt_image")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "templates" / "index.html"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"
GENERATED_DIR = BASE_DIR / "generated"
ENV_FILE = BASE_DIR / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                remote_url TEXT,
                local_url TEXT NOT NULL,
                local_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (generation_id) REFERENCES generations(id) ON DELETE CASCADE
            );
            """
        )


load_env_file(ENV_FILE)
init_storage()
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "high"
    n: int = Field(default=1, ge=1, le=4)
    transparent: bool = False


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY manquante. Configure-la dans le fichier .env.",
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://build.lewisnote.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_openai_error_payload(exc: OpenAIError) -> dict[str, object]:
    payload: dict[str, object] = {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }

    for key in ("status_code", "code", "param", "type", "request_id"):
        value = getattr(exc, key, None)
        if value is not None:
            payload[key] = value

    body = getattr(exc, "body", None)
    if body is not None:
        payload["body"] = body

    response = getattr(exc, "response", None)
    if response is not None:
        payload["response_status_code"] = getattr(response, "status_code", None)
        try:
            response_text = response.text
        except Exception:
            response_text = None

        if response_text:
            payload["response_text"] = response_text[:4000]

    return payload


def create_generation(prompt: str) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO generations (prompt, status, created_at) VALUES (?, ?, ?)",
            (prompt, "processing", now_iso()),
        )
        return int(cursor.lastrowid)


def update_generation_status(
    generation_id: int, status: str, error_message: str | None = None
) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE generations SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, generation_id),
        )


def save_image_from_remote(
    remote_url: str, generation_id: int, index: int
) -> tuple[str, str]:
    filename = (
        f"gen-{generation_id}-{index}-{int(datetime.now().timestamp() * 1000)}.png"
    )
    local_path = GENERATED_DIR / filename

    with urllib.request.urlopen(remote_url, timeout=120) as response:
        content = response.read()

    local_path.write_bytes(content)
    local_url = f"/generated/{filename}"
    return str(local_path), local_url


def save_image_from_b64(
    encoded_image: str, generation_id: int, index: int
) -> tuple[str, str]:
    filename = (
        f"gen-{generation_id}-{index}-{int(datetime.now().timestamp() * 1000)}.png"
    )
    local_path = GENERATED_DIR / filename

    try:
        content = base64.b64decode(encoded_image)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Image b64 invalide.") from exc

    local_path.write_bytes(content)
    local_url = f"/generated/{filename}"
    return str(local_path), local_url


def store_image(
    generation_id: int, remote_url: str | None, local_url: str, local_path: str
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO images (generation_id, remote_url, local_url, local_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (generation_id, remote_url, local_url, local_path, now_iso()),
        )


def get_history(limit: int) -> list[dict[str, str | int | None]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                g.id,
                g.prompt,
                g.status,
                g.created_at,
                (
                    SELECT i.local_url
                    FROM images i
                    WHERE i.generation_id = g.id
                    ORDER BY i.id ASC
                    LIMIT 1
                ) AS cover_local_url,
                (
                    SELECT i.remote_url
                    FROM images i
                    WHERE i.generation_id = g.id
                    ORDER BY i.id ASC
                    LIMIT 1
                ) AS cover_remote_url,
                (
                    SELECT COUNT(*)
                    FROM images i
                    WHERE i.generation_id = g.id
                ) AS image_count
            FROM generations g
            ORDER BY g.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history: list[dict[str, str | int | None]] = []
    for row in rows:
        cover_url = row["cover_local_url"] or row["cover_remote_url"]
        history.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "status": row["status"],
                "created_at": row["created_at"],
                "cover_url": cover_url,
                "image_count": row["image_count"],
            }
        )

    return history


def get_generation_detail(generation_id: int) -> dict:
    with get_db() as conn:
        generation = conn.execute(
            "SELECT id, prompt, status, error_message, created_at FROM generations WHERE id = ?",
            (generation_id,),
        ).fetchone()

        if generation is None:
            raise HTTPException(status_code=404, detail="Generation introuvable.")

        image_rows = conn.execute(
            """
            SELECT id, remote_url, local_url, local_path, created_at
            FROM images
            WHERE generation_id = ?
            ORDER BY id ASC
            """,
            (generation_id,),
        ).fetchall()

    images = [
        {
            "id": row["id"],
            "remote_url": row["remote_url"],
            "local_url": row["local_url"],
            "local_path": row["local_path"],
            "created_at": row["created_at"],
        }
        for row in image_rows
    ]

    return {
        "id": generation["id"],
        "prompt": generation["prompt"],
        "status": generation["status"],
        "error_message": generation["error_message"],
        "created_at": generation["created_at"],
        "images": images,
    }


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=500, detail="Frontend introuvable.")
    return FileResponse(FRONTEND_FILE)


@app.get("/api/history")
def history(limit: int = Query(default=30, ge=1, le=200)) -> dict[str, list[dict]]:
    return {"items": get_history(limit=limit)}


@app.get("/api/history/{generation_id}")
def history_detail(generation_id: int) -> dict:
    return get_generation_detail(generation_id)


@app.post("/api/generate")
def generate_images(payload: ImageGenerationRequest) -> dict:
    generation_id = create_generation(prompt=payload.prompt)

    request: dict[str, str | int] = {
        "model": "gpt-image-2",
        "prompt": payload.prompt,
        "size": payload.size,
        "quality": payload.quality,
        "n": payload.n,
    }

    if payload.transparent:
        request["background"] = "transparent"
        request["output_format"] = "png"

    try:
        client = get_client()
        response = client.images.generate(**request)

        output_images: list[dict[str, str | None]] = []
        for index, item in enumerate(response.data, start=1):
            remote_url = getattr(item, "url", None)
            b64_json = getattr(item, "b64_json", None)

            if remote_url:
                local_path, local_url = save_image_from_remote(
                    remote_url=remote_url,
                    generation_id=generation_id,
                    index=index,
                )
            elif b64_json:
                local_path, local_url = save_image_from_b64(
                    encoded_image=b64_json,
                    generation_id=generation_id,
                    index=index,
                )
            else:
                continue

            store_image(
                generation_id=generation_id,
                remote_url=remote_url,
                local_url=local_url,
                local_path=local_path,
            )
            output_images.append(
                {
                    "local_url": local_url,
                    "remote_url": remote_url,
                }
            )

        if not output_images:
            raise HTTPException(
                status_code=502,
                detail="Aucune image exploitable retournee par le provider.",
            )

        update_generation_status(generation_id=generation_id, status="completed")
        return {
            "id": generation_id,
            "prompt": payload.prompt,
            "status": "completed",
            "images": output_images,
        }
    except HTTPException as exc:
        update_generation_status(
            generation_id=generation_id,
            status="failed",
            error_message=exc.detail,
        )
        raise
    except OpenAIError as exc:
        error_payload = get_openai_error_payload(exc)
        logger.exception(
            "Echec requete gpt-image-2: %s",
            json.dumps(error_payload, ensure_ascii=False, default=str),
        )
        update_generation_status(
            generation_id=generation_id,
            status="failed",
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=502, detail=f"Erreur provider image: {exc}"
        ) from exc
    except Exception as exc:
        update_generation_status(
            generation_id=generation_id,
            status="failed",
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Erreur interne: {exc}") from exc
