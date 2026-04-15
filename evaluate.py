from pathlib import Path
from search import SearchEngine
import time
# queries that I think will work well as the give enough information without being long
GOOD_QUERIES = [
    "cristina lopes",
    "machine learning research",
    "uci informatics faculty",
    "computer networks research",
    "acm publications",
    "ics graduate program",
    "data mining research",
    "uci artificial intelligence",
    "informatics faculty",
    "computer science department"
]
# queries that I think will work poorly as they are low in information or contain stop words
BAD_QUERIES = [
    "ai",
    "faculty",
    "software engineering",
    "courses",
    "security",
    "systems",
    "research",
    "students",
    "program",
    "master of software engineering"
]

def run_evaluation(engine, queries, label, f):
    f.write(f"\n===== {label} =====\n\n")

    for q in queries:
        f.write(f"Query: {q}\n")

        # START TIMER
        start = time.time()

        results = engine.search_and_rank(q, top_k=5)

        # END TIMER
        end = time.time()
        runtime = end - start

        f.write(f"Runtime: {runtime:.4f} seconds\n")

        if not results:
            f.write("No results\n\n")
            continue

        for i, (score, doc_id, url) in enumerate(results, 1):
            f.write(f"{i}. {url}\n")

        f.write("\n")

def main():
    base_dir = Path(__file__).resolve().parent
    engine = SearchEngine(base_dir)

    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        run_evaluation(engine, GOOD_QUERIES, "GOOD QUERIES", f)
        run_evaluation(engine, BAD_QUERIES, "BAD QUERIES", f)

    print("Evaluation written to evaluation_results.txt")


if __name__ == "__main__":
    main()