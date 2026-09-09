"""
app.py
======
Run with:
    uvicorn backend.app:app --reload --port 8000

Then open http://localhost:8000 in a browser.

Before first use, load a round's queue:
    python -m backend.queue_loader --round 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config as C
from backend import db
from backend.routes import plants, verdicts

app = FastAPI(title="tp_qa Review App")

db.init_db()

app.include_router(plants.router, prefix="/api/plants", tags=["plants"])
app.include_router(verdicts.router, prefix="/api/verdict", tags=["verdicts"])

app.mount("/static", StaticFiles(directory=str(C.FRONTEND_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(C.FRONTEND_DIR / "index.html"))
