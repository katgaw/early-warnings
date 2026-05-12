from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_SAVE_DIR = Path("models") / "all-MiniLM-L6-v2"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
if not (BASE_SAVE_DIR / "modules.json").is_file():
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(str(BASE_SAVE_DIR))
print("Base model at:", BASE_SAVE_DIR.resolve())