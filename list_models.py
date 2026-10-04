"""List the Gemini text models your API key can use.

Listing models does NOT count against your generation quota.
Run:  python list_models.py
Then copy the names you want into config.yaml (models.generation_fallbacks / models.hyde).
"""
import config  # noqa: F401  (loads .env)
from google import genai

client = genai.Client()
names = sorted(
    m.name.removeprefix("models/")
    for m in client.models.list()
    if "generateContent" in (getattr(m, "supported_actions", None) or [])
)
print("Models that can generate text with your key:\n")
for n in names:
    print("  ", n)
