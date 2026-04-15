from pathlib import Path
import struct

BASE_DIR = Path(__file__).resolve().parent.parent  # go up from search-engine/
INDEX_DIR = BASE_DIR / "index_dir"

def read_postings(offset, length, post_path=INDEX_DIR / "inverted_indx.bin"):
    postings = []
    with open(post_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)

    pos = 0
    doc_freq = struct.unpack_from(">I", data, pos)[0]
    pos += 4

    for _ in range(doc_freq):
        doc_id, freq, importance = struct.unpack_from(">III", data, pos)
        postings.append((doc_id, freq, importance))
        pos += 12

    return postings


def load_terms_index(idx_path=INDEX_DIR / "terms.idx"):
    term_dict = {}

    with open(idx_path, "rb") as f:
        while True:
            length_bytes = f.read(2)
            if not length_bytes:
                break

            term_len = struct.unpack(">H", length_bytes)[0]
            term = f.read(term_len).decode("utf-8")
            offset = struct.unpack(">Q", f.read(8))[0]
            plen   = struct.unpack(">I", f.read(4))[0]

            term_dict[term] = (offset, plen)

    return term_dict


def print_first_100_terms_with_postings(limit=100):
    term_dict = load_terms_index()
    sorted_terms = sorted(term_dict.keys())

    for i, term in enumerate(sorted_terms[:limit], 1):
        offset, length = term_dict[term]
        postings = read_postings(offset, length)

        print(f"{i}. term='{term}'")
        if len(postings) < 20:
            print(f"   offset={offset}, length={length}")
            print(f"   postings={postings}\n")


if __name__ == "__main__":
    print_first_100_terms_with_postings()
