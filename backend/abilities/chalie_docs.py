from typing import ClassVar

from abilities._ability import Ability
from services.file_mapper_service import FileMapperService

_VERSION_FILE = FileMapperService.get_version_path()

_QUERY_URLS: dict[str, list[str]] = {
    "basics": [
        "https://chalie.ai/guide/getting-started/",
        "https://chalie.ai/how-it-works/",
    ],
    "tools": [
        "https://chalie.ai/guide/getting-started/",
    ],
    "releases": [
        "https://chalie.ai/releases/",
    ],
    "code-base": [
        "https://github.com/chalie-ai/chalie",
    ],
}


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except OSError:
        return "unknown"


class ChalieDocsAbility(Ability):
    def get_name(self) -> str:
        return "chalie_docs"

    def get_summary(self) -> str:
        return "Look up Chalie's own documentation — what it is, its tools, release history, or codebase."

    def get_examples(self) -> list[str]:
        return [
            "what is chalie",
            "how does chalie work",
            "what tools does chalie have",
            "show me the latest chalie release notes",
            "where is the chalie source code",
            "tell me about chalie's capabilities",
        ]

    def get_search_tooltip(self) -> str:
        return "chalie documentation and self-reference"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "enum": ["basics", "tools", "releases", "code-base"],
                "description": (
                    "basics — what Chalie is and how it works. "
                    "tools — available tools and capabilities. "
                    "releases — version history and changelogs. "
                    "code-base — source code repository."
                ),
            },
        },
        "required": ["query"],
    }

    def run(self, params: dict) -> dict:
        query = params.get("query", "").strip().lower()
        urls = _QUERY_URLS.get(query)
        if not urls:
            return {"error": f"Unknown query '{query}'. Use one of: {', '.join(_QUERY_URLS)}"}
        version = _read_version()
        joined = " & ".join(urls)
        return {"text": f"To learn about Chalie ({version}) {query} use the read tool and visit: {joined}"}
