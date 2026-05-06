"""Build the plant RAG index.

Reads all 5 tables from plant.duckdb and chunks any files in
data/plant_knowledge/ (txt, md, pdf), embeds everything with
text-embedding-3-small, and saves the index to data/plant_rag/.

Run from the garden/ directory:
    cd garden
    python build/plant_rag/main.py
"""

import json
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(Path(__file__).parents[3] / ".env")


# ── Parameters ────────────────────────────────────────────────────────────────

GARDEN_ROOT   = Path(__file__).parents[2]
DATA_DIR      = GARDEN_ROOT / "data"
PLANT_DB_PATH = str(DATA_DIR / "plant.duckdb")
KNOWLEDGE_DIR = DATA_DIR / "plant_knowledge"
OUTPUT_DIR    = DATA_DIR / "plant_rag"

EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
EMBED_BATCH   = 100


# ── Plant DB document builder ─────────────────────────────────────────────────

def build_plant_documents(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Build one rich text document per plant type combining all 5 tables."""
    plant_types = conn.execute(
        "SELECT plant_type_id, name, category FROM plant_types ORDER BY name"
    ).fetchall()

    docs = []
    for plant_type_id, name, category in plant_types:
        lines = [f"=== {name} ===", f"Category: {category}"]

        variety = conn.execute(
            """SELECT sun_tolerance, water_required, spacing_inches,
                      height_inches_estimate, days_to_harvest,
                      temp_min_air_f, temp_min_ground_f,
                      soil_n, soil_p, soil_k,
                      growth_needs, post_harvest_soil_needs,
                      outdoor_sow_date_range, harvest_timing
               FROM plant_varieties WHERE plant_type_id = ? LIMIT 1""",
            [plant_type_id],
        ).fetchone()

        if variety:
            lines.append("\nCare:")
            lines.append(f"  Sun: {variety[0]} | Water: {variety[1]}")
            if variety[2]:
                lines.append(f"  Spacing: {variety[2]} inches")
            if variety[3]:
                lines.append(f"  Height: {variety[3]} inches")
            if variety[4]:
                lines.append(f"  Days to harvest: {variety[4]}")
            if variety[5] or variety[6]:
                lines.append(f"  Temperature min: air {variety[5]}°F, ground {variety[6]}°F")
            if any(x is not None for x in [variety[7], variety[8], variety[9]]):
                lines.append(f"  Soil NPK: N={variety[7]}, P={variety[8]}, K={variety[9]}")
            if variety[10]:
                lines.append(f"  Growth notes: {variety[10]}")
            if variety[11]:
                lines.append(f"  Post-harvest soil: {variety[11]}")
            if variety[12]:
                lines.append(f"  Outdoor sow: {variety[12]}")
            if variety[13]:
                lines.append(f"  Harvest timing: {variety[13]}")

        companions = conn.execute(
            """SELECT companion_name, relationship FROM plant_companions
               WHERE plant_type_id = ? ORDER BY relationship, companion_name""",
            [plant_type_id],
        ).fetchall()
        if companions:
            good = [c[0] for c in companions if c[1] == "companion"]
            bad  = [c[0] for c in companions if c[1] == "antagonist"]
            if good:
                lines.append(f"\nCompanion plants: {', '.join(good)}")
            if bad:
                lines.append(f"Antagonist plants: {', '.join(bad)}")

        pests = conn.execute(
            "SELECT pest_name, symptoms, treatment FROM plant_pests WHERE plant_type_id = ?",
            [plant_type_id],
        ).fetchall()
        if pests:
            lines.append("\nPests:")
            for pest_name, symptoms, treatment in pests:
                lines.append(
                    f"  - {pest_name}: {symptoms or 'see treatment'}. "
                    f"Treatment: {treatment or 'n/a'}."
                )

        diseases = conn.execute(
            "SELECT disease_name, symptoms, treatment FROM plant_diseases WHERE plant_type_id = ?",
            [plant_type_id],
        ).fetchall()
        if diseases:
            lines.append("\nDiseases:")
            for disease_name, symptoms, treatment in diseases:
                lines.append(
                    f"  - {disease_name}: {symptoms or 'see treatment'}. "
                    f"Treatment: {treatment or 'n/a'}."
                )

        docs.append({
            "text": "\n".join(lines),
            "source": f"plant_db:{name}",
            "plant_name": name,
        })

    return docs


# ── File chunker ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_knowledge_files(knowledge_dir: Path) -> list[dict]:
    """Load and chunk .txt, .md, and .pdf files from knowledge_dir."""
    if not knowledge_dir.exists():
        return []

    chunks = []
    for path in sorted(knowledge_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            file_chunks = _chunk_text(text, source=path.name)
            chunks.extend(file_chunks)
            print(f"  {path.name}: {len(text):,} chars → {len(file_chunks)} chunks")
        elif suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                file_chunks = _chunk_text(text, source=path.name)
                chunks.extend(file_chunks)
                print(f"  {path.name}: {len(reader.pages)} pages → {len(file_chunks)} chunks")
            except Exception as exc:
                print(f"  Skipped {path.name}: {exc}")

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_documents(docs: list[dict], client: OpenAI) -> np.ndarray:
    """
    Embed all document texts in batches using text-embedding-3-small.

    Returns a float32 array of shape (N, 1536) with L2-normalised rows
    so cosine similarity reduces to a dot product at query time.
    """
    texts = [d["text"] for d in docs]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp  = client.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend(e.embedding for e in resp.data)
        print(f"  Embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")

    arr   = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    """Build and save the plant RAG index."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    print("Reading plant.duckdb ...")
    conn = duckdb.connect(PLANT_DB_PATH, read_only=True)
    plant_docs = build_plant_documents(conn)
    conn.close()
    print(f"  Built {len(plant_docs)} plant documents")

    print("Loading knowledge files ...")
    file_chunks = load_knowledge_files(KNOWLEDGE_DIR)
    print(f"  Loaded {len(file_chunks)} chunks from files")

    all_docs = plant_docs + file_chunks
    print(f"\nTotal documents to embed: {len(all_docs)}")

    print(f"Embedding with {EMBED_MODEL} ...")
    embeddings = embed_documents(all_docs, client)

    emb_path = OUTPUT_DIR / "embeddings.npy"
    doc_path = OUTPUT_DIR / "docs.json"
    np.save(str(emb_path), embeddings)
    with open(doc_path, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"\nDone — saved {len(all_docs)} documents to {OUTPUT_DIR}/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  docs.json:      {len(all_docs)} entries")


if __name__ == "__main__":
    run()
