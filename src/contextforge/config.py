SUPPORTED_EXTENSIONS = {
    ".py",
    ".go",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".txt",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
}

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".go": "go",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".sql": "sql",
    ".txt": "text",
}

DEFAULT_CHUNK_SIZE = 20
DEFAULT_CHUNK_OVERLAP = 5
PROJECT_NAME_ALLOWED_PATTERN = "^[a-zA-Z0-9_-]+$"