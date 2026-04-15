import argparse
import math
import re
import time
import struct
from pathlib import Path
from typing import Dict, List, Tuple
import mmap

from functools import lru_cache
from nltk.stem import PorterStemmer

# moving the porter stemmer object to module scope to avoid re-creating
ps = PorterStemmer()


# 1) TOKENIZATION + STEMMING (NOTE: MUST MATCH INDEXER)
# Important: Search query must be processed the same way
# as documents were processed during indexing in M1.
#
# If we stem in indexing but not in search -> never find matches.
# Example: "running" becomes "run". If we query "running" and don't stem,
# you'll look for "running" but the index only contains "run".
#
# This regex is the same one used indexer:
# It extracts only sequences of letters/numbers (no punctuation).
# Example: "UCI's!" -> tokens: ["uci", "s"]
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+\b")


def get_tokens(text: str) -> List[str]:
    """
    Convert a string into a list of tokens (words).

    - lower() makes it case-insensitive
    - regex finds alphanumeric tokens only
    """
    return _TOKEN_RE.findall(text.lower())


def stemming(token_list: List[str]) -> List[str]:
    """
    Apply Porter stemming to each token.

    Porter stemming reduces words to their "root".
    Example:
        "running" -> "run"
        "software" -> "softwar"
    """
    return [ps.stem(w) for w in token_list]


def normalize_query(q: str) -> List[str]:
    """
       Full query normalization pipeline:
       - tokenize query
       - apply Porter stemming
       - generate bigram tokens to support phrase matching
       """
    tokens = stemming(get_tokens(q))

    # generate bigrams
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(tokens[i] + "_" + tokens[i + 1])

    tokens.extend(bigrams)

    return tokens


# 2) DISK INDEX READING (DEVELOPER VERSION REQUIREMENT)
# Developer version:
# - allowed to load the "dictionary" (term -> where postings live)
# - Not allowed to load all postings into memory
#
# Index is split into two files:
#
# (A) terms.idx: compact lookup table mapping term -> (offset, length)
# (B) inverted_indx.bi: actual postings lists in binary format
#
# Format details (must match indexer merge writing):
#
# terms.idx stores entries like:
# term_len (2 bytes, unsigned short, big-endian)
# term bytes (utf-8)
# offset (8 bytes, unsigned long long) into inverted_indx.bin
# length (4 bytes, unsigned int): how many bytes to read
#
# inverted_indx.bin stores the posting list blob at that offset:
# doc_freq (4 bytes): number of postings
# repeated doc_freq times:
#   doc_id (4 bytes), tf (4 bytes), importance (4 bytes)
#
# This allows: seek directly to postings for a term and read only that.


def load_terms_index(terms_idx_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Load the dictionary mapping:
        term -> (offset_in_bin_file, length_in_bytes)

    This is small enough to keep fully in memory.
    """
    term_dict: Dict[str, Tuple[int, int]] = {}

    # open in binary mode because terms.idx is NOT text
    with terms_idx_path.open("rb") as f:
        while True:
            # read 2 bytes: term length
            length_bytes = f.read(2)

            # if we hit EOF (end of file), stop
            if not length_bytes:
                break

            # unpack bytes into integer (big-endian unsigned short)
            term_len = struct.unpack(">H", length_bytes)[0]

            # read the term string bytes and decode as utf-8
            term = f.read(term_len).decode("utf-8")

            # read offset (8 bytes) and postings blob length (4 bytes)
            offset = struct.unpack(">Q", f.read(8))[0]
            plen = struct.unpack(">I", f.read(4))[0]

            term_dict[term] = (offset, plen)

    return term_dict


def read_postings(mm: mmap.mmap, offset: int, length: int) -> List[Tuple[int, int, int]]:
    """
    Read one term's postings list from inverted_indx.bin.

    We do not load the whole file.
    We only:
        - seek to offset
        - read exactly 'length' bytes

    Returns:
        List of (doc_id, tf, importance)
    """
    # read the binary blob for this term from the memory-mapped file
    data = mm[offset : offset + length]

    # Now we parse the binary blob from the in-memory bytes data.
    pos = 0

    # First 4 bytes = doc_freq (# of postings)
    doc_freq = struct.unpack_from(">I", data, pos)[0]
    pos += 4

    postings: List[Tuple[int, int, int]] = []

    # Each posting entry is 12 bytes total:
    #   doc_id (4) + tf (4) + importance (4)
    for _ in range(doc_freq):
        doc_id, tf, importance = struct.unpack_from(">III", data, pos)
        postings.append((doc_id, tf, importance))
        pos += 12

    # postings should already be sorted by doc_id because indexer sorted them
    return postings


# 3) DOC ID -> URL MAPPING
# Index stores docIDs (integers), not urls, for space efficiency.
# Need to convert docID results back into URLs for printing.
#
# docID_mapping.txt format:
# 0: https://example.com/page1
# 1: https://example.com/page2


def load_docid_mapping(mapping_path: Path) -> Dict[int, str]:
    """
    Load docID_mapping.txt into memory:
        doc_id -> url

    This is safe memory-wise (just one line per doc).
    """
    m: Dict[int, str] = {}

    with mapping_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # expected format: "id: url"
            try:
                left, right = line.split(":", 1)
                doc_id = int(left.strip())
                url = right.strip()
                if url:
                    m[doc_id] = url
            except ValueError:
                # if a line doesn't match format, ignore it
                continue

    return m


# 4) BOOLEAN AND INTERSECTION
# Developer version requires at minimum: AND queries.
#
# AND means:
# query = "machine learning"
# doc must contain both "machine" AND "learning"
#
# We do AND by intersecting the doc_id lists from each term.
#
# Since postings lists are sorted by doc_id, we can do intersection
# efficiently in linear time using two pointers.


def intersect_sorted_ids(a: List[int], b: List[int]) -> List[int]:
    """
    Intersection of two sorted lists using two pointers.

    Example:
        a = [1, 2, 4, 10]
        b = [2, 4, 7, 10]
        result = [2, 4, 10]
    """
    i = j = 0
    out: List[int] = []

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return out


# 5) SCORING / RANKING (TF-IDF + importance boost)
#
# TF-IDF basics:
# tf = term frequency in document (how many times term appears)
# df = document frequency (how many docs contain the term)
# N  = total documents
#
# idf is higher if the term is rare across documents.
# tf is higher if the term appears more often in the document.
#
# importance is M1 "boost" based on HTML tags (title/h1/bold).


def tf_weight(tf: int) -> float:
    """
    Term Frequency weighting.
    We use log scaling to reduce the impact of huge tf values.
    """
    return 1.0 + math.log(tf)


def idf_weight(N: int, df: int) -> float:
    """
    Inverse Document Frequency weighting.
    smoothed to avoid division by zero:
        log((N+1)/(df+1)) + 1
    """
    return math.log((N + 1) / (df + 1)) + 1.0


def importance_boost(importance: int) -> float:
    """
    Boost documents where the term occurred in important places like:
      - title
      - headings
      - bold text

    Keep it small so it doesn't dominate tf-idf.
    """
    return 1.0 + 0.4 * importance


# 6) SEARCH ENGINE CLASS 

class SearchEngine:
    """
    SearchEngine ties everything together:
      - loads term dictionary (terms.idx)
      - loads docID->URL mapping
      - reads postings from inverted_indx.bin on demand
      - AND-intersects postings lists
      - ranks results
    """

    def __init__(self, base_dir: Path):
        # base_dir is where search.py lives
        self.base_dir = base_dir

        # index files are stored in base_dir/index_dir/
        self.index_dir = base_dir / "index_dir"
        self.terms_idx_path = self.index_dir / "terms.idx"
        self.inv_bin_path = self.index_dir / "inverted_indx.bin"
        self.mapping_path = base_dir / "docID_mapping.txt"

        # open the inv_bin_path file once and keep for all reads
        f = self.inv_bin_path.open("rb")
        self.inv_bin_file = f
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # basic safety checks: fail early if files are missing
        if not self.terms_idx_path.exists():
            raise FileNotFoundError(f"Missing {self.terms_idx_path}. Run indexer.py first.")
        if not self.inv_bin_path.exists():
            raise FileNotFoundError(f"Missing {self.inv_bin_path}. Run indexer.py first.")
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Missing {self.mapping_path}. Run indexer.py first.")

        # Load term dictionary: term -> (offset, length)
        self.term_dict = load_terms_index(self.terms_idx_path)

        # Load docID mapping: doc_id -> url
        self.docid_to_url = load_docid_mapping(self.mapping_path)

        # N = total documents (needed for IDF)
        self.N = len(self.docid_to_url)

    @lru_cache(maxsize=1000)
    def get_term_postings(self, term: str) -> List[Tuple[int, int, int]]:
        """
        Given a term, return its postings list from disk.
        If term doesn't exist in dictionary, return empty.
        """
        meta = self.term_dict.get(term)
        if meta is None:
            return []

        offset, length = meta
        return read_postings(self.mm, offset, length)

    def search_and_rank(self, query: str, top_k: int = 5, debug: bool = False) -> List[Tuple[float, int, str]]:
        """
        Main search function.

        Steps:
          1) normalize query (tokenize + stem)
          2) fetch postings for each term from disk
          3) AND-intersect doc_id lists
          4) score candidates using tf-idf + importance
          5) return top_k results
        """
        # normalize query the same way documents were normalized
        terms = normalize_query(query)

        # Remove duplicate terms (e.g., "uci uci uci") but keep order.
        seen = set()
        uniq_terms = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                uniq_terms.append(t)
        terms = uniq_terms

        if debug:
            print(f"[debug] raw query: {query}")
            print(f"[debug] normalized terms: {terms}")

        # if query is empty after normalization, return no results
        if not terms:
            return []
        
        # compute the DFs for each term (used for getting next term to drop
        # in rolling query)
        term_df = {}
        for term in terms:
            postings = self.get_term_postings(term)
            term_df[term] = len(postings)

        # copy of terms to use in while loop
        current_terms = terms.copy()

        # growing list of matched docs
        results = []
        # keep track of seen docs
        seen_docs = set()

        while current_terms:
            if debug:
                print(f"[debug] querying on subset: {current_terms}")
                
            sub_results = self._search_on_terms(current_terms, top_k, debug)

            for score, doc_id, url in sub_results:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    results.append((score, doc_id, url))
            
            if len(results) >= top_k:
                results.sort(key=lambda x: x[0], reverse=True)
                return results[:top_k]
            
            # get the next term to drop
            drop_term = min(current_terms, key=lambda x: term_df.get(x, 0))

            if debug:
                print(f"[debug] drop_term: {drop_term}")

            # remove drop_term from current_terms
            current_terms.remove(drop_term)
        
        # return whatever we found even if there is not at least k documents
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _search_on_terms(self, terms: List[str], top_k: int, debug: bool) -> List[Tuple[float, int, str]]:
        # Store term metadata in these structures:
        #
        # term_postings[t] = [(doc_id, tf, importance), ...]
        # term_doc_ids[t]  = [doc_id1, doc_id2, ...]  (for intersection)
        # term_post_dict[t] = {doc_id: (tf, importance)} (for scoring fast)
        # term_df[t] = how many docs contain t 
        term_doc_ids: Dict[str, List[int]] = {}
        term_post_dict: Dict[str, Dict[int, Tuple[int, int]]] = {}
        term_df: Dict[str, int] = {}

        # Load postings for each query term
        for t in terms:
            postings = self.get_term_postings(t)
            df = len(postings)
            term_df[t] = df

            if debug:
                print(f"[debug] term='{t}' df={df}")

            # AND short-circuit:
            # If ANY term doesn't exist in index, then no doc can contain all terms.
            if df == 0:
                return []

            # Extract doc_ids only (still sorted)
            term_doc_ids[t] = [doc_id for (doc_id, tf, imp) in postings]

            # Store tf + importance per doc for scoring
            term_post_dict[t] = {doc_id: (tf, imp) for (doc_id, tf, imp) in postings}

        # AND intersection across all term doc lists
        # Optimization: intersect smallest df first (fewer candidates sooner)
        terms_by_df = sorted(terms, key=lambda x: term_df[x])

        candidates = term_doc_ids[terms_by_df[0]]
        for t in terms_by_df[1:]:
            candidates = intersect_sorted_ids(candidates, term_doc_ids[t])
            if not candidates:
                return []

        if debug:
            print(f"[debug] candidates after AND: {len(candidates)}")

        # multi-term queries get a small boost
        phrase_bonus = 1.0
        if len(terms) > 1:
            phrase_bonus = 1.2

        # Score each candidate document
        results: List[Tuple[float, int, str]] = []
        for doc_id in candidates:
            score = 0.0

            # Add up each term's contribution to the doc score
            for t in terms:
                tf, imp = term_post_dict[t][doc_id]  # must exist due to AND match

                score += (
                        tf_weight(tf) *
                        idf_weight(self.N, term_df[t]) *
                        importance_boost(imp) *
                        phrase_bonus
                )
            # normalize score a little to avoid very long pages dominating
            score = score / (1 + math.log(len(terms)))

            # Convert doc_id to URL for display
            url = self.docid_to_url.get(doc_id, "")
            results.append((score, doc_id, url))

        # sort best score first
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def close_inv_bin_file(self):
        """
        Close the inverted index binary file when done.
        """
        self.mm.close()
        self.inv_bin_file.close()



# 7) COMMANDS:
#   1) python search.py
#   2) required queries: python search.py --run-required

REQUIRED_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]


def main():
    parser = argparse.ArgumentParser(
        description="M2 Search Component (developer version) - AND retrieval + TF-IDF ranking"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Project base directory (defaults to the folder containing this script).",
    )
    parser.add_argument("--topk", type=int, default=5, help="How many results to print.")
    parser.add_argument("--debug", action="store_true", help="Print debug info.")
    parser.add_argument(
        "--run-required",
        action="store_true",
        help="Run the 4 required M2 queries and print top 5 URLs for each.",
    )
    args = parser.parse_args()

    # If you didn't pass --base-dir, we assume search.py is in the project root
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    # Create the engine (loads dictionary + mapping)
    print("Loading index...")
    engine = SearchEngine(base_dir)

    # If we're running the required queries, print them all and exit
    if args.run_required:
        for q in REQUIRED_QUERIES:
            print("\n" + "=" * 60)
            print(f"Query: {q}")

            results = engine.search_and_rank(q, top_k=args.topk, debug=args.debug)

            if not results:
                print("No results.")
                continue

            # Only print urls (since report asks for top urls)
            for i, (score, doc_id, url) in enumerate(results, 1):
                print(f"{i}. {url}")

        print("\n" + "=" * 60)
        engine.close_inv_bin_file()  # close file when done
        return

    # Otherwise interactive mode:
    print("Search Engine (M2) — type a query, or 'quit' to exit.")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            engine.close_inv_bin_file()  # close file when done
            break

        start_time = time.perf_counter()
        results = engine.search_and_rank(q, top_k=args.topk, debug=args.debug)
        end_time = time.perf_counter()

        if not results:
            print("No results.")
            continue
        print("Top 5 results:")
        for i, (score, doc_id, url) in enumerate(results, 1):
            print(f"{i}. {url} ")

        elapsed = (end_time - start_time) * 1000  # convert to milliseconds
        print(f"Search took {elapsed:.2f} ms.")

    print("\nExiting...")

if __name__ == "__main__":
    main()