const API_BASE = "/api";

async function login() {
    const username = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
    const error = document.getElementById('error');
    
    if(!username || !pass) { error.innerText = "All fields are required"; return; }
    
    try {
        const res = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: username, password: pass})
        });
        const data = await res.json();
        
        if(data.success) {
            localStorage.setItem('full_name', data.full_name);
            window.location.href = '/chat';
        } else {
            error.innerText = data.message;
        }
    } catch(e) { error.innerText = "Server Error"; }
}

async function signup() {
    const full = document.getElementById('fullname').value;
    const user = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
    const error = document.getElementById('error');
    
    if(!full || !user || !pass) { error.innerText = "All fields are required"; return; }
    
    try {
        const res = await fetch(`${API_BASE}/signup`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: user, password: pass, full_name: full})
        });
        const data = await res.json();
        
        if(data.success) {
            alert("Account created! Please login.");
            window.location.href = '/';
        } else {
            error.innerText = data.message;
        }
    } catch(e) { error.innerText = "Server Error"; }
}

function logout() {
    localStorage.removeItem('full_name');
    window.location.href = '/';
}

function handleKey(e) {
    if(e.key === 'Enter') sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const msg = input.value.trim();
    if(!msg) return;
    
    // Add User Bubble
    addBubble(msg, 'user');
    input.value = '';
    
    // Show Loading
    const loadingId = addBubble("Processing...", 'bot', true);
    
    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        
        // Remove loading
        document.getElementById(loadingId).remove();
        
        // Show Bot Response
        addBubble(data.response, 'bot');
        
    } catch(e) {
        document.getElementById(loadingId).innerText = "Error connecting to server.";
    }
}

function addBubble(text, type, isLoading=false) {
    const box = document.getElementById('chat-box');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerText = text;
    if(isLoading) {
        const id = 'loading-' + Date.now();
        div.id = id;
        return id;
    }
    div.appendChild(bubble);
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}
