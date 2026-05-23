"""
Generate search routing embeddings directly into providers.sqlite.

Reads all provider examples, embeds them using EmbeddingService, and writes
the vectors into an ``example_embeddings`` vec0 virtual table inside the same
providers.sqlite database.

Run from the backend directory:
    cd backend && python -m utils.generate_search_cache
"""

import sqlite3
import sys

from services.embedding_service import EmbeddingService
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

_DB_PATH = FileMapperService.get_search_providers_db_path()


def main():
    if not _DB_PATH.exists():
        print(f"ERROR: {_DB_PATH} not found")
        sys.exit(1)

    print(f"Reading examples from {_DB_PATH}")

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception:
            conn.load_extension('vec0')

        examples = conn.execute(
            "SELECT id, example_query FROM provider_examples ORDER BY id"
        ).fetchall()

        if not examples:
            print("ERROR: no rows found in provider_examples")
            sys.exit(1)

        print(f"Found {len(examples)} examples — loading embedding model...")
        emb_service = EmbeddingService()

        # Drop existing embeddings table if present (idempotent rebuild)
        conn.execute("DROP TABLE IF EXISTS example_embeddings")
        conn.commit()
        conn.execute("CREATE VIRTUAL TABLE example_embeddings USING vec0(embedding float[768])")
        conn.commit()

        print("Embedding examples in batches of 32...")

        texts = [row[1] for row in examples]
        ids = [row[0] for row in examples]

        batch_size = 32
        inserted = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            embeddings = emb_service.generate_embeddings_batch(batch_texts)

            for example_id, embedding in zip(batch_ids, embeddings):
                blob = pack_embedding(embedding)
                conn.execute(
                    "INSERT INTO example_embeddings(rowid, embedding) VALUES (?, ?)",
                    (example_id, blob),
                )
                inserted += 1

            conn.commit()

            if inserted % 100 == 0 or inserted == len(examples):
                print(f"  {inserted}/{len(examples)} embedded...")
    finally:
        conn.close()

    size_kb = _DB_PATH.stat().st_size / 1024
    print(f"\nDone. {inserted} examples embedded into {_DB_PATH.name}")
    print(f"Database size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
