from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psycopg2
import psycopg2.extras


DIRECTORY = Path(__file__).resolve().parent
# --- DATABASE SETUP ---

DB_PARAMS: dict = {
    "host":   "localhost",
    "port":   5432,
    "dbname": "topology",
    "user":   "postgres",
    # Password via ~/.pgpass
}

# --- FASTAPI SETUP ---

app = FastAPI(title="IndoorGML 3DGS Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    # Compulsory headers indicated in mkkellogg's GitHub.
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"]   = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


def db_conn():
    return psycopg2.connect(**DB_PARAMS, cursor_factory=psycopg2.extras.RealDictCursor)


@app.get("/api/scene-data")
async def scene_data():
    """
    Grab all rooms, states, and transitions from PostGIS.
    Geometry is returned as GeoJSON.
    """
    try:
        conn = db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, room_name, cell_type,
                       ST_AsGeoJSON(geom)::json AS geom,
                       height, file_path, transform
                FROM room
                ORDER BY id
            """)
            rooms = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT id, room_id, ST_AsGeoJSON(geom)::json AS geom
                FROM state
                ORDER BY id
            """)
            states = [dict(s) for s in cur.fetchall()]

            cur.execute("""
                SELECT t.id, t.from_state, t.to_state,
                       ST_AsGeoJSON(t.geom)::json AS geom,
                       t.dual_boundary, t.door_height,
                       s1.room_id AS from_room,
                       s2.room_id AS to_room
                FROM transition t
                JOIN state s1 ON s1.id = t.from_state
                JOIN state s2 ON s2.id = t.to_state
                ORDER BY t.id
            """)
            transitions = [dict(t) for t in cur.fetchall()]

        conn.close()
        return JSONResponse({"rooms": rooms, "states": states, "transitions": transitions})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


app.mount("/static", StaticFiles(directory="static"), name="static")

with open(DIRECTORY / "viewer.html", "r", encoding="utf-8") as f:
    VIEWER_HTML = f.read()
    
@app.get("/", response_class=HTMLResponse)
async def root():
    return VIEWER_HTML

# --- MAIN ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)