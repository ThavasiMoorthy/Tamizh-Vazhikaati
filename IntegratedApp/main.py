from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
import os

# Import our engines
from backend.kb_engine import KBEngine
from backend.text_gen_engine import TextGenEngine
from backend.poem_gen_engine import PoemGenEngine
from backend import database
import logging

# Configure diagnostic logging
log_file = os.path.join(os.getcwd(), "api_debug.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

app = FastAPI(title="Tamil Kovil Vazhikati")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize Engines
kb_engine = KBEngine(
    kb_path="../QA/kb_updated.json", 
    alias_path="../QA/alias_map (3).json",
    nl_path="../QA/nl_questions_map.json"
)
# text_engine = TextGenEngine() # Disabled for KB-only
# poem_engine = PoemGenEngine() # Disabled for KB-only

# Initialize DB
database.init_db()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    print("Starting up engines...")
    print("Engines Ready (AI Models: DISABLED).")
    
    # Start Ngrok Tunnel only if not in production and requested
    if os.environ.get("USE_NGROK") == "true":
        try:
            public_url = ngrok.connect(8000)
            print(f"\n[Ngrok] PUBLIC URL: {public_url}")
            print(f"[Ngrok] Swagger UI: {public_url}/docs\n")
            
            # Save to a file for easy access
            with open("../ngrok_url.txt", "w") as f:
                f.write(str(public_url))
        except Exception as e:
            print(f"[Ngrok] Error starting tunnel: {e}")
    else:
        print("Skipping Ngrok startup (USE_NGROK not set to 'true').")

# --- Auth Models ---
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str
    full_name: str

class ChatRequest(BaseModel):
    message: str
    mode: str = "auto"

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return FileResponse("templates/react_index.html")

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/login")
async def login(data: LoginRequest):
    print(f"Login attempt: {data.username}")
    user = database.get_user(data.username)
    if user and user.password == data.password:
        return {"success": True, "full_name": user.full_name}
    return {"success": False, "message": "Invalid credentials"}

@app.post("/api/signup")
async def signup(data: SignupRequest):
    print(f"Signup attempt: {data.username}")
    user = database.User(username=data.username, password=data.password, full_name=data.full_name)
    if database.create_user(user):
        return {"success": True}
    return {"success": False, "message": "Username already exists"}

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    try:
        msg = data.message.strip()
        
        # KB-Only Mode
        kb_answer = kb_engine.get_answer(msg)
        
        if kb_answer:
            response_text = kb_answer
            source = "kb"
        else:
            response_text = "மன்னிக்கவும், உங்கள் கேள்விக்கான பதில் என்னிடம் இல்லை. (KB-Only Mode)"
            source = "system"
            
        logging.info(f"Query: '{msg}' -> Response: '{response_text[:100]}...' (Source: {source})")
        return {"response": response_text, "source": source}
    except Exception as e:
        import traceback
        error_msg = f"Server Error: {str(e)}"
        print(traceback.format_exc())
        return {"response": error_msg, "source": "error"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
