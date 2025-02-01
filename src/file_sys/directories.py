from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent
assert REPO_DIR.is_dir()

__all__ = ['REPO_DIR']
