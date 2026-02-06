import torch
import torch.nn as nn
import sentencepiece as spm
import math
import os
import zipfile
import tempfile

# ==========================================
# 1. Model Architecture Definitions
# ==========================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin()[None, None, :, :], emb.cos()[None, None, :, :]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        sin, cos = self.rope(T, x.device)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# ==========================================
# 2. Engine Class
# ==========================================

class PoemGenEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 256 # Hyperparameter

    def load_model(self):
        from huggingface_hub import hf_hub_download
        
        # Use Hugging Face model (complete and working)
        repo_id = "athiathiathi/tamil_poem_generator"
        
        print(f"[Poem Gen] Loading model from Hugging Face: {repo_id}")
        print(f"[Poem Gen] Device: {self.device}")

        try:
            # Download files from Hugging Face
            print(f"[Poem Gen] Downloading tokenizer...")
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tamil_sp_model5.model")
            
            print(f"[Poem Gen] Downloading model checkpoint...")
            model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
            
            # Load Tokenizer
            print(f"[Poem Gen] Loading tokenizer...")
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            self.tokenizer = sp
            print(f"[Poem Gen] Tokenizer loaded. Vocab size: {sp.get_piece_size()}")

            # Initialize Model Architecture
            VOCAB_SIZE = 9000
            D_MODEL = 384
            N_HEADS = 8
            N_LAYERS = 6
            MAX_LEN = self.max_len

            print(f"[Poem Gen] Initializing model architecture...")
            self.model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN)

            # Load checkpoint
            print(f"[Poem Gen] Loading model weights...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if hasattr(checkpoint, "state_dict"):
                self.model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                # If checkpoint IS the model
                self.model = checkpoint

            self.model.to(self.device)
            self.model.eval()
            print("[Poem Gen] Model loaded successfully from Hugging Face!")
            
        except Exception as e:
            print(f"[Poem Gen] ERROR loading model: {e}")
            raise

    def generate(self, subject, theme="General", max_tokens=150, temperature=0.75):
        if not self.model or not self.tokenizer:
            return "Error: Poem Model not loaded."

        prompt = f"""<துவக்கம்>
<வழிமுறை>
கீழ்க்கண்ட தகவல்களை அடிப்படையாகக் கொண்டு
ஒரு புதுக்கவிதையை எழுதவும்.
</வழிமுறை>
<பொருள்> {subject}
<கருப்பொருள்> {theme}
"""
        try:
            sp = self.tokenizer
            device = self.device
            model = self.model
            
            ids = sp.encode(prompt, out_type=int)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                for _ in range(max_tokens):
                    x_cond = x[:, -self.max_len:]
                    logits = model(x_cond)
                    logits = logits[:, -1, :] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, next_id), dim=1)
                    
                    decoded = sp.decode(x[0].tolist())
                    if "<முடிவு>" in decoded:
                        break
            
            # Post-processing: Extract only the poem content
            # The model often echoes the prompt. We need to find the content after the metadata.
            # Common tags: <துவக்கம்>, <வழிமுறை>, <பொருள்>, <கருப்பொருள்>, <குறிச்சொற்கள்>, <முடிவு>
            
            # Start by removing the prompt part if it exists
            final_content = decoded
            
            # If the model echoes everything, the poem is usually after the metadata tags.
            # We'll look for the last occurrence of these metadata headers.
            headers = ["<கருப்பொருள்>", "<பொருள்>", "<குறிச்சொற்கள்>"]
            last_pos = -1
            for header in headers:
                pos = final_content.rfind(header)
                if pos != -1:
                    # Move past the header AND any immediate single line of metadata
                    line_end = final_content.find("\n", pos)
                    if line_end != -1:
                        last_pos = max(last_pos, line_end)
                    else:
                        last_pos = max(last_pos, pos + len(header))
            
            if last_pos != -1:
                # Take everything after the last metadata header
                poem = final_content[last_pos:].strip()
            else:
                # Fallback: just strip known prompt tags if found
                poem = final_content
                for tag in ["<துவக்கம்>", "<வழிமுறை>", "</வழிமுறை>", "<பொருள்>", "<கருப்பொருள்>", "<குறிச்சொற்கள்>"]:
                    poem = poem.replace(tag, "")

            # Cleanup
            poem = poem.replace("<முடிவு>", "").strip()
            
            # If extraction resulted in something too short or empty, return the whole thing minus tags
            if len(poem) < 10:
                poem = decoded
                for tag in ["<துவக்கம்>", "<வழிமுறை>", "</வழிமுறை>", "<பொருள்>", "<கருப்பொருள்>", "<குறிச்சொற்கள்>", "<முடிவு>"]:
                    poem = poem.replace(tag, "")
                poem = poem.strip()

            return poem
        except Exception as e:
            return f"Error generating poem: {e}"
