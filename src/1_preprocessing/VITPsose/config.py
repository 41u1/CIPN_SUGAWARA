from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

MOVIE_DIR = DATA_DIR / "movie" / "target"
OUTPUT_MOVIE_DIR = OUTPUT_DIR / "movie"
OUTPUT_RAW_DIR = OUTPUT_DIR / "raw"