try:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
    import sentencepiece as spm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Text Gen] Warning: Torch/Transformers not found. Text generation will be disabled.")
    # Dummy classes to prevent NameErrors if classes are referenced
    class nn: Module = object
    class PreTrainedModel: pass
    class PretrainedConfig: pass
    class PreTrainedTokenizer: pass

import contextlib
import os

# ---------------- CONFIG ----------------
# Use LOCAL model bundle instead of HuggingFace repo
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "temple_model_bundle_20260201")
MODEL_FILENAME = "tamil_clm_15m_instruction.pt"
TOKENIZER_FILENAME = "tamil_clm.model"

# ---------------- MODEL DEFINITIONS ----------------
class TamilCLMConfig(PretrainedConfig):
    model_type = "tamil-clm"
    def __init__(self, vocab_size=8000, max_position_embeddings=512, hidden_size=384, num_hidden_layers=6, num_attention_heads=6, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, -1e9).softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TamilCLMForCausalLM(PreTrainedModel):
    config_class = TamilCLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.token = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_attention_heads) for _ in range(config.num_hidden_layers)])
        self.ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.token(input_ids) + self.pos(pos)
        for block in self.blocks: x = block(x)
        x = self.ln(x)
        return self.lm_head(x)

# ✅ STANDALONE TOKENIZER (No HuggingFace inheritance to avoid conflicts)
class TamilSentencePieceTokenizer:
    def __init__(self, vocab_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_file)
        
        # Define special tokens
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        
        # Get token IDs
        self.eos_token_id = self.sp.piece_to_id("</s>")
        self.bos_token_id = self.sp.piece_to_id("<s>")
        self.unk_token_id = self.sp.piece_to_id("<unk>")
        self.pad_token_id = self.sp.piece_to_id("<pad>")
    
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
    
    def encode(self, text):
        """Encode text to token IDs"""
        return self.sp.encode(text, out_type=int)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        return self.sp.decode(ids)

class TextGenEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.load_error = None

    def load_model(self):
        if not TORCH_AVAILABLE:
            print("[Text Gen] Torch not available. Skipping load.")
            self.load_error = "Torch not available. (VC++ Redist Missing?)"
            return

        print("[Text Gen] Loading model and tokenizer from LOCAL bundle...")
        try:
            model_weights_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            sp_model_path = os.path.join(MODEL_DIR, TOKENIZER_FILENAME)
            
            print(f"[Text Gen] Model path: {model_weights_path}")
            print(f"[Text Gen] Tokenizer path: {sp_model_path}")
            
            # Load tokenizer
            self.tokenizer = TamilSentencePieceTokenizer(vocab_file=sp_model_path)
            
            # Load model
            config = TamilCLMConfig()
            self.model = TamilCLMForCausalLM(config)
            
            # Load weights
            state_dict = torch.load(model_weights_path, map_location="cpu")
            
            # Handle key name differences
            if "head.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict.pop("head.weight")
                state_dict["lm_head.bias"] = state_dict.pop("head.bias")
                
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # ✅ CRITICAL: Verify vocab sizes match
            assert self.tokenizer.vocab_size == self.model.config.vocab_size, \
                f"Vocab mismatch: tokenizer={self.tokenizer.vocab_size}, model={self.model.config.vocab_size}"
            
            self.loaded = True
            print(f"[Text Gen] Model loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"[Text Gen] Failed to load model: {e}")
            import traceback
            print(traceback.format_exc())
            self.load_error = str(e)

    def generate(self, question: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        if not self.loaded:
            if not TORCH_AVAILABLE:
                return "Text Model Unavailable (Torch Library Missing on Server)"
            
            return f"Error: Text Model not loaded. Reason: {self.load_error}"
            
        prompt = f"கேள்வி: {question}\nபதில்:"
        try:
            # ✅ Use direct SentencePiece encoding
            ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long)
            
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)

            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    next_token_logits = outputs[0, -1, :] / temperature
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # ✅ Properly check EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
        
            # ✅ Use direct SentencePiece decoding
            full_text = self.tokenizer.decode(input_ids[0].tolist())
            answer = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
            return answer
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error Generating Text: {str(e)}"
