from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uvicorn
import os

# =================CONFIGURATION=================
# Ensure these files are in the same directory as this script
KB_FILE = "kb_updated.json"
ALIAS_FILE = "alias_map.json"  # Rename your 'alias_map (3).json' to this
# ===============================================

app = FastAPI(title="Tamil Temple QA Server")

# Add CORS middleware to allow connections from your mobile app/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
kb_data = {}
alias_data = {}

def load_data():
    """Load the Knowledge Base and Alias Map from JSON files."""
    global kb_data, alias_data
    try:
        if not os.path.exists(KB_FILE):
            print(f"CRITICAL ERROR: {KB_FILE} not found!")
            return
        if not os.path.exists(ALIAS_FILE):
            print(f"CRITICAL ERROR: {ALIAS_FILE} not found!")
            return

        print(f"Loading Knowledge Base from {KB_FILE}...")
        with open(KB_FILE, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            
        print(f"Loading Alias Map from {ALIAS_FILE}...")
        with open(ALIAS_FILE, "r", encoding="utf-8") as f:
            alias_data = json.load(f)
            
        print("Data loaded successfully.")
        print(f"KB Entries: {len(kb_data)}")
        print(f"Alias Entries: {len(alias_data)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")

# Load data on startup
@app.on_event("startup")
async def startup_event():
    load_data()

# Data Models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    found: bool

# API Endpoints
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to answer questions based on exact match in the Alias Map.
    Payload: {"question": "your question here"}
    """
    user_question = request.question.strip()
    
    # 1. Look up the question in the Alias Map
    canonical_key = alias_data.get(user_question)
    
    if not canonical_key:
        return AnswerResponse(
            answer="மன்னிக்கவும், இந்த கேள்விக்கான பதில் என்னிடம் இல்லை. (Answer not found)",
            found=False
        )
    
    # 2. Retrieve data from Knowledge Base using the canonical key
    kb_entry = kb_data.get(canonical_key)
    
    if not kb_entry:
        return AnswerResponse(
            answer="தகவல் காணப்படவில்லை. (Data missing in KB)",
            found=False
        )
    
    # 3. Format the answer
    if isinstance(kb_entry, dict):
        answers = [str(v) for v in kb_entry.values() if v]
        final_answer = "\n".join(answers)
    else:
        final_answer = str(kb_entry)
        
    return AnswerResponse(answer=final_answer, found=True)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Simple Web Interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tamil Temple QA</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f0f2f5; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #1a1a1a; text-align: center; margin-bottom: 30px; }
            input[type="text"] { width: 100%; padding: 15px; font-size: 16px; border: 2px solid #e0e0e0; border-radius: 8px; box-sizing: border-box; transition: border-color 0.3s; }
            input[type="text"]:focus { border-color: #4CAF50; outline: none; }
            button { width: 100%; padding: 15px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 15px; transition: background-color 0.3s; }
            button:hover { background-color: #45a049; }
            #result { margin-top: 25px; padding: 20px; border-radius: 8px; background-color: #fafafa; border: 1px solid #eee; min-height: 60px; line-height: 1.6; white-space: pre-wrap; }
            .error { color: #d32f2f; }
            .success { color: #2e7d32; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tamil Temple QA Bot</h1>
            <input type="text" id="question" placeholder="Enter your question here (e.g., அண்ணாமலையார் கோயில் எந்த மாவட்டத்தில் உள்ளது?)...">
            <button onclick="askQuestion()">Ask Question</button>
            <div id="result">Answer will appear here...</div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const resultDiv = document.getElementById('result');
                if (!question) return;

                resultDiv.innerHTML = 'Searching...';
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question }),
                    });
                    const data = await response.json();
                    if (data.found) {
                        resultDiv.innerHTML = '<span class="success">' + data.answer + '</span>';
                    } else {
                        resultDiv.innerHTML = '<span class="error">' + data.answer + '</span>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<span class="error">Error connecting to server.</span>';
                }
            }
            document.getElementById("question").addEventListener("keyup", function(event) {
                if (event.keyCode === 13) askQuestion();
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
