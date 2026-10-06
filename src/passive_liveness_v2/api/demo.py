"""Demo frontend route: serves the single-page Capture Deck UI."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


@router.get("/", include_in_schema=False)
def index(request: Request):
    """The demo frontend: live camera capture + liveness verdicts."""
    return templates.TemplateResponse(request, "index.html")
