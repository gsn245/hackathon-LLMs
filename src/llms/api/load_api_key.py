from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def load_api_key() -> str:
    """Load API key from file."""
    if not (ROOT_DIR / "the-key").exists():
        raise ValueError(
            "Ask Miguel for the the-key, and place the-key on the root of the project."
        )

    with open(ROOT_DIR / "the-key", "r") as f:
        return f.read().strip()
