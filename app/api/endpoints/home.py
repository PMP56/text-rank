from fastapi import APIRouter, Request, Form
from fastapi.responses import FileResponse
from app.api.deps import template
from models.TextRank import TextRank

router = APIRouter()
textrank = TextRank()

@router.get("/graph")
def get_graph():
    image_path = "app/static/graph.png"  # Update this to the actual path
    return FileResponse(image_path)

@router.get('/')
async def home(request: Request):
    context = {
        "request": request,
        "algorithm": "summarization",  # Set the default algorithm
        "original": "",
        "output": ""
    }

    return template("index.html", context)

@router.post("/")
async def submit(
    request: Request,
    algorithm: str = Form(...),
    original: str = Form(""),
    keyword_count: int = Form(10),
    sentence_count: int = Form(3),
):
    
    output = ""
    error = None
    
    if (len(original) < 10):
        return template("index.html", {
            "request": request,
            "algorithm": algorithm,
            "original": original,
            "output": output,
            "error": "Text to small"
        })

    try:
        if (algorithm == "summarization"):
            output = textrank.summarize(original, sentence_count=sentence_count)
        else:
            keywords = textrank.keyword(original, plot=True, keyword_count=keyword_count)[0]
            keywords = "\n".join([k[0] for  k in keywords])
            output = keywords
    except Exception as e:
        error = str(e)

    context = {
        "request": request,
        "algorithm": algorithm,
        "original": original,
        "output": output,
        "error": error,
        "graph": True if algorithm == "keywords" else False,
    }
    
    return template("index.html", context)