import sys
import os

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.kb_engine import KBEngine

def test_kb():
    print(f"Current Working Directory: {os.getcwd()}")
    
    # These are the paths used in main.py
    kb_path = "../kb_updated.json"
    alias_path = "../alias_map.json"
    
    print(f"Testing paths:")
    print(f"KB Path: {os.path.abspath(kb_path)}")
    print(f"Alias Path: {os.path.abspath(alias_path)}")
    
    if not os.path.exists(kb_path):
        print("ERROR: KB File not found at relative path!")
    else:
        print("SUCCESS: KB File found.")
        
    if not os.path.exists(alias_path):
        print("ERROR: Alias File not found at relative path!")
    else:
        print("SUCCESS: Alias File found.")

    engine = KBEngine(kb_path, alias_path)
    
    # Test specific question
    test_q = "அண்ணாமலையார் கோயில் அமைந்துள்ள இடம் எது?"
    print(f"\nTesting Question: '{test_q}'")
    
    answer = engine.get_answer(test_q)
    print(f"Result: {answer}")
    
    if answer:
        print("KB Engine is WORKING.")
    else:
        print("KB Engine returned None. Check keys.")
        # Debug alias content if failed
        if test_q in engine.alias_data:
            print(f"Key exists in alias data. Mapped to: {engine.alias_data[test_q]}")
        else:
            print("Key does NOT match any entry in alias data.")

if __name__ == "__main__":
    test_kb()
