from fastapi import APIRouter, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from app.api.deps import template
from models.TextRank import TextRank

router = APIRouter()
textrank = TextRank()

# In-memory cache: stores last graph data so /graph-data can serve it
_last_graph_data: dict = {"nodes": [], "edges": []}


@router.get("/graph")
def get_graph():
    image_path = "app/static/graph.png"
    return FileResponse(image_path)


@router.get("/graph-data")
def get_graph_data():
    """Return the last-computed keyword graph as JSON for interactive rendering."""
    return JSONResponse(_last_graph_data)


@router.get("/")
async def home(request: Request):
    context = {
        "request": request,
        "algorithm": "summarization",
        "original": "",
        "output": "",
        "keyword_count": 10,
        "sentence_count": 3,
        "epoch": 10,
        "damping_factor": 0.85,
    }
    return template("index.html", context)


@router.post("/")
async def submit(
    request: Request,
    algorithm: str = Form(...),
    original: str = Form(""),
    keyword_count: int = Form(10),
    sentence_count: int = Form(3),
    epoch: int = Form(10),
    damping_factor: float = Form(0.85),
):
    global _last_graph_data

    output = ""
    error = None
    show_graph = False

    if len(original) < 10:
        return template("index.html", {
            "request": request,
            "algorithm": algorithm,
            "original": original,
            "output": "",
            "error": "Text too small",
            "keyword_count": keyword_count,
            "sentence_count": sentence_count,
            "epoch": epoch,
            "damping_factor": damping_factor,
        })

    try:
        if algorithm == "summarization":
            output = textrank.summarize(original, sentence_count=sentence_count)
        else:
            graph_data = textrank.get_graph_data(
                original,
                keyword_count=keyword_count,
                epoch=epoch,
                damping_factor=damping_factor,
            )
            _last_graph_data = graph_data
            output = "\n".join(node["label"] for node in graph_data["nodes"])
            show_graph = True

    except Exception as e:
        error = str(e)

    context = {
        "request": request,
        "algorithm": algorithm,
        "original": original,
        "output": output,
        "error": error,
        "graph": show_graph,
        "keyword_count": keyword_count,
        "sentence_count": sentence_count,
        "epoch": epoch,
        "damping_factor": damping_factor,
    }

    return template("index.html", context)