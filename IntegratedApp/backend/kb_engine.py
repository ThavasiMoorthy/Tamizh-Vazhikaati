import json
import os

class KBEngine:
    def __init__(self, kb_path: str, alias_path: str, nl_path: str = None):
        self.kb_path = kb_path
        self.alias_path = alias_path
        self.nl_path = nl_path
        self.kb_data = {}
        self.alias_data = {}
        self.nl_data = {}
        self.load_data()

    def load_data(self):
        try:
            if os.path.exists(self.kb_path):
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    self.kb_data = json.load(f)
            if os.path.exists(self.alias_path):
                with open(self.alias_path, "r", encoding="utf-8") as f:
                    self.alias_data = json.load(f)
            if self.nl_path and os.path.exists(self.nl_path):
                with open(self.nl_path, "r", encoding="utf-8") as f:
                    self.nl_data = json.load(f)
            print(f"[KB Engine] Loaded {len(self.kb_data)} KB entries, {len(self.alias_data)} aliases, and {len(self.nl_data)} NL questions.")
        except Exception as e:
            print(f"[KB Engine] Error loading data: {e}")

    def get_answer(self, question: str) -> str | None:
        """Returns the answer if found, otherwise None."""
        q = question.strip()
        
        # 1. Exact NL Question Match
        if q in self.nl_data:
            alias_key = self.nl_data[q]
            if alias_key in self.alias_data:
                canonical_key = self.alias_data[alias_key]
                return self._format_entry(self.kb_data.get(canonical_key))
            # Fallback if the alias_key from NL map is the canonical key itself
            if alias_key in self.kb_data:
                return self._format_entry(self.kb_data.get(alias_key))

        # 2. Exact Alias Match
        if q in self.alias_data:
            canonical_key = self.alias_data[q]
            return self._format_entry(self.kb_data.get(canonical_key))

        # 2. Normalized Match (remove extra spaces)
        # TODO: Advanced normalization if needed
        
        # 3. Fuzzy/Keyword Search (Fallback)
        # Log for debugging if needed (via api_debug.log in main.py)
        
        # Sort keys by length descending to find the most specific match first
        all_keys = sorted(self.alias_data.keys(), key=len, reverse=True)
        
        for key in all_keys:
            # Check if key is in query (User: "About Annamalaiyar Temple" -> Key: "Annamalaiyar")
            # OR if query is in key (User: "Meenakshi" -> Key: "Madurai Meenakshi Amman")
            if (key in q or q in key) and len(key) > 3 and len(q) > 3:
                canonical_key = self.alias_data[key]
                return self._format_entry(self.kb_data.get(canonical_key))
                
        return None

    def _format_entry(self, kb_entry):
        if not kb_entry: return None
        if isinstance(kb_entry, dict):
            answers = [str(v) for v in kb_entry.values() if v]
            return "\n".join(answers)
        else:
            return str(kb_entry)
