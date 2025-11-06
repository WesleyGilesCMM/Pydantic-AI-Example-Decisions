from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates

load_dotenv(override=True)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.models import create_db
from fastapi.responses import HTMLResponse

create_db()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


from api.routes import *


app.include_router(auth_router)
app.include_router(decision_router)

templates = Jinja2Templates(directory="api/templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/present", response_class=HTMLResponse)
async def present(request: Request):
    return templates.TemplateResponse("presentation.html", {"request": request})