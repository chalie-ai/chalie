"""
Seed additional provider_examples into search_tool_providers.sqlite.

Run from the backend directory after inserting rows:
    cd backend && python -m utils.seed_routing_examples
    cd backend && python -m utils.generate_search_cache

Uses INSERT OR IGNORE so this script is idempotent — safe to re-run.
Requires a UNIQUE constraint on (provider_id, example_query); the script
creates it if missing.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.file_mapper_service import FileMapperService  # noqa: E402

_DB_PATH = FileMapperService.get_search_providers_db_path()

# wikipedia provider_id = 1
_WIKIPEDIA_EXAMPLES = [
    "most visited tourist attractions in the world",
    "popular destinations in europe",
    "what is the most famous mountain",
    "which national park gets the most visitors",
    "what is the most photographed building",
    "most popular hiking trail in the world",
    "what is the busiest airport in the world",
    "which museum has the most visitors",
    "most famous landmark in japan",
    "best known temple in india",
    "what tourist site is most visited",
    "which european capital is most popular",
    "most well known beach in the world",
    "what is the most iconic skyscraper",
    "world's most popular natural wonder",
    "most recognized monument in france",
    "what is the most visited city in asia",
    "top tourist destination in south america",
    "most famous waterfall on earth",
    "what is the world's most visited theme park",
    "most popular lake for tourists",
    "which country attracts the most visitors",
    "most famous historic site in greece",
    "what is the most photographed bridge",
    "best known ancient ruin",
    "most popular mountain to climb",
    "what are the most visited islands in the world",
    "famous canyon visited by millions",
    "most recognized building on earth",
    "what city has the most famous skyline",
]

# nominatim provider_id = 12
_NOMINATIM_EXAMPLES = [
    "where is the most visited beach in hawaii",
    "location of the most photographed lighthouse",
    "where is the world's most visited waterfall",
    "where is the busiest tourist street in paris",
    "location of the most popular mountain resort",
    "where is the most famous market in marrakech",
    "where is the most visited national park in the usa",
    "location of the most crowded tourist square in prague",
    "where is the most popular hot spring in iceland",
    "address of the most famous castle in scotland",
    "where is the world's most visited beach resort",
    "location of the most visited pilgrimage site",
    "where is the largest outdoor market in africa",
    "location of the most popular harbor viewpoint in sydney",
    "where is the most photographed street in london",
    "where is the most visited ancient temple in cambodia",
    "location of the world's most popular ski resort",
    "where is the busiest street market in tokyo",
]


def main() -> None:
    if not _DB_PATH.exists():
        print(f"ERROR: {_DB_PATH} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        # Ensure uniqueness guard exists (idempotent DDL)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_examples "
            "ON provider_examples(provider_id, example_query)"
        )
        conn.commit()

        # Resolve provider IDs dynamically — never hard-code them
        rows = conn.execute("SELECT id, name FROM providers").fetchall()
        id_by_name = {name: pid for pid, name in rows}

        wikipedia_id = id_by_name.get("wikipedia")
        nominatim_id = id_by_name.get("nominatim")

        if wikipedia_id is None or nominatim_id is None:
            print(f"ERROR: expected providers 'wikipedia' and 'nominatim', found: {list(id_by_name)}")
            sys.exit(1)

        inserted = 0
        for text in _WIKIPEDIA_EXAMPLES:
            cur = conn.execute(
                "INSERT OR IGNORE INTO provider_examples (provider_id, example_query) VALUES (?, ?)",
                (wikipedia_id, text),
            )
            inserted += cur.rowcount

        for text in _NOMINATIM_EXAMPLES:
            cur = conn.execute(
                "INSERT OR IGNORE INTO provider_examples (provider_id, example_query) VALUES (?, ?)",
                (nominatim_id, text),
            )
            inserted += cur.rowcount

        conn.commit()

        total_wiki = conn.execute(
            "SELECT COUNT(*) FROM provider_examples WHERE provider_id = ?", (wikipedia_id,)
        ).fetchone()[0]
        total_nom = conn.execute(
            "SELECT COUNT(*) FROM provider_examples WHERE provider_id = ?", (nominatim_id,)
        ).fetchone()[0]

        print(f"Inserted {inserted} new rows (OR IGNORE skips duplicates).")
        print(f"  wikipedia total: {total_wiki}")
        print(f"  nominatim total: {total_nom}")
        print("\nNext step: cd backend && python -m utils.generate_search_cache")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
