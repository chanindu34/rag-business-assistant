import os
import sys

# Make the project root importable when pytest runs from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
