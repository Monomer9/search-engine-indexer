import json
from pathlib import Path
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
import os
from collections import defaultdict
import heapq
import math
import time
import struct
import hashlib

class Posting:
    """
    Class to represent a posting. The doc_id of a url.
    """
    def __init__(self, doc_id, freq = 0, importance = 0):
        self.doc_id = doc_id
        self.freq = freq
        self.importance = importance


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.doc_count = 0
        self.doc_id_mapping = {}

    def update_index(self, term_data: defaultdict[str, list[int]], url: str) -> None:
        self.doc_count += 1
        doc_id = self.doc_count

        # term_data stores per-document statistics:
        # each word appears once with its total count and how "important" it was on the page
        for term, (freq, importance) in term_data.items():
            # get the postings list for the term
            # if term not seen before → create empty list
            postings = self.index.setdefault(term, [])

            # create a posting representing this document
            posting = Posting(doc_id)

            # store how many times the word appears in this doc
            posting.freq = freq

            # importance tracks whether the word appeared in title/headings/bold,
            # which will later boost ranking during search
            posting.importance = importance

            # add posting to the inverted index
            postings.append(posting)


        # search engine will later print URLs, not doc_ids
        self.doc_id_mapping[doc_id] = url

    def save_index(self, filename: str = "inverted_index.json") -> None:
        # convert the in-memory index (Python objects) into plain dictionaries
        # so it can be written to disk as JSON
        serializable = {}

        for term, postings in self.index.items():
            # turn Posting objects into simple data so JSON can store it
            serializable[term] = [
                {"doc_id": p.doc_id, "tf": p.freq, "importance": p.importance}
                for p in postings
            ]
        # write the inverted index to disk so it can be loaded later by the search program
        with open(filename, "w") as f:
            json.dump(serializable, f)

    def flush_partial(self, part_num: int, out_dir: Path) -> Path:
        out_path = out_dir / f"partial_index_{part_num}.json"
        with out_path.open("w") as f:
            for term, postings in sorted(self.index.items()):
                # turn Posting objects into simple data so JSON can store it
                serializable = { term: [ {"doc_id": p.doc_id, "tf": p.freq, "importance": p.importance} for p in postings ] }
                f.write(json.dumps(serializable) + "\n")

        return out_path

# produces a generator object when called so that we don't have to store whole files locally
def stream_partial(path: Path):
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            term = next(iter(obj.keys()))
            postings = obj[term]
            yield term, postings


def merge_indx_files(partial_indx_paths: list[Path], terms_indx_path: Path, inv_indx_path: Path) -> int:
    """
    returns the number of terms in the final inverted index
    """
    term_count = 0
    # call stream_partial to create the generator for each partial index file
    streams = [stream_partial(path) for path in partial_indx_paths]
    # need heap for efficient processing
    heap = []

    # initialize the heap with the first term from each file
    for sid, stream in enumerate(streams):
        try:
            # gets the next line which is the first item
            term, postings = next(stream)
            # pushes the term onto the heap
            heapq.heappush(heap, (term, sid, postings))
        except StopIteration:
            # occurs when there are no more terms to generate
            pass
    
    with open(terms_indx_path, "wb") as terms_out, open(inv_indx_path, "wb") as indx_out:

        # while the heap is not empty
        while heap:
            # pop an item of the heap
            term, sid, postings = heapq.heappop(heap)
            merged = postings

            # while there are same terms in the heap with a seperate posting list
            while heap and heap[0][0] == term:
                _, other_sid, other_postings = heapq.heappop(heap)
                merged.extend(other_postings)

                # advance the generator to get next term for file that just had
                # term postings merged
                try:
                    new_term, new_postings = next(streams[other_sid])
                    heapq.heappush(heap, (new_term, other_sid, new_postings))
                except StopIteration:
                    # occurs when there are no more terms to generate
                    pass
            
            # sort the Postings list by the doc_id
            merged.sort(key=lambda x: x["doc_id"])

            # write term to the final inverted index
            
            # get the current byte position
            offset = indx_out.tell()

            # write how many postings for the term to indx
            indx_out.write(struct.pack(">I", len(merged)))

            # write each posting as (doc_id, term frequency, importance) to file
            for posting in merged:
                indx_out.write(struct.pack(">III", posting["doc_id"], posting["tf"], posting["importance"]))

            # get the length of the posting list
            length = indx_out.tell() - offset

            # write term mapping (term: (offset, length)) to terms file
            term_as_bytes = term.encode("utf-8")
            terms_out.write(struct.pack(">H", len(term_as_bytes))) # store the length of the term so that can load into memory in search
            terms_out.write(term_as_bytes) # stores term
            terms_out.write(struct.pack(">Q", offset)) # stores offset to find postings list
            terms_out.write(struct.pack(">I", length)) # stores length of postings list

            #increment term count
            term_count += 1

            # advance the original generator of the current iteration
            try:
                new_term, new_postings = next(streams[sid])
                heapq.heappush(heap, (new_term, sid, new_postings))
            except StopIteration:
                # occurs when there are no more terms to generate
                pass
        
    return term_count


# tokenizer logic
def get_tokens(text: str) -> list:
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

def stemming(token_list: list) -> list:
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(word) for word in token_list]

def parse_json(path: Path) -> tuple[defaultdict[list[int]], str]:
    global seen_pages
    term_data = defaultdict(lambda: [0, 0])  # [freq, importance]

    def process_text(text, weight):
            tokens = stemming(get_tokens(text))
################### EXTRA CREDIT: BIGRAMS ###############################
            # add bigrams so machine learning -> machine_learning
            bigrams = []
            for i in range(len(tokens) - 1):
                bigrams.append(tokens[i] + "_" + tokens[i + 1])

            tokens.extend(bigrams)
            for t in tokens:
                term_data[t][0] += 1          # frequency
                term_data[t][1] += weight     # importance

    with open(path, "r") as f:
        data = json.load(f)
        soup = BeautifulSoup(data["content"], "lxml")

        # remove non-important words
        for spam in soup(["script", "style", "meta"]):
            spam.decompose()

        # extracts title
        if soup.title:
            process_text(soup.title.get_text(), 3)
            soup.title.extract()

        # extracts headings
        for tag in soup.find_all(["h1","h2","h3"]):
            process_text(tag.get_text(), 2)
            tag.extract()

        # extracts bold
        for tag in soup.find_all(["b","strong"]):
            process_text(tag.get_text(), 1)
            tag.extract()

#################### EXTRA CREDIT: ANCHOR TEXT INDEXING ##################################
        for a in soup.find_all("a"):
            anchor_text = a.get_text()
            process_text(anchor_text, 2)  # treat as moderately important

        text = soup.get_text()

################# EXTRA CREDIT: EXACT DUPLICATE DETECTION ############################
        # compute an MD5 hash of page text to detect identical pages
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        # skips duplicate pages
        if content_hash in seen_pages:
            return None, None

        seen_pages.add(content_hash)

        # normal text processing
        process_text(text, 0)

        return term_data, data['url']

# probably need to import someone's tokenizer in here (tokenize on alphanumeric)
if __name__ == '__main__':
    start = time.perf_counter()
    base = Path("developer/DEV")
    index_dir = Path("index_dir")
    index_dir.mkdir(exist_ok=True)

    inverted_indx = InvertedIndex()
    # keep track of all the partial index files
    index_file_list = []
    index_file_indx = 0
    seen_pages = set() # stores hashes of previously seen pages (duplicate detection)

    for i, json_f in enumerate(base.rglob("*.json"), 1):
        term_data, url = parse_json(json_f)
        # do exact duplicate detection
        if term_data is None:
            continue

        inverted_indx.update_index(term_data, url)

        if i % 500 == 0:
            print(f"Indexed {i} documents...")

        if i % 5000 == 0:
            # saves index for 2500 documents
            indx_file = inverted_indx.flush_partial(index_file_indx, index_dir)
            # adds the new partial index to list of index files
            index_file_list.append(indx_file)
            # clears index to make room in memory
            inverted_indx.index.clear()
            index_file_indx += 1
            print(f"Flushed index at {i} documents.")

    
    # adds any remaining indexed items
    if inverted_indx.index:
        indx_file = inverted_indx.flush_partial(index_file_indx, index_dir)
        index_file_list.append(indx_file)
    
    # merge all the index files
    unique_tokens = merge_indx_files(index_file_list, index_dir/"terms.idx", index_dir/"inverted_indx.bin")

    end = time.perf_counter()

    #inverted_indx.save_index("../index_output/inverted_index.json")
    with open("report.txt", "w") as f:
        f.write("\n===== INDEX STATS =====\n")
        f.write(f"Documents: {inverted_indx.doc_count}\n")
        f.write(f"Unique tokens: {unique_tokens}\n")
        f.write(f"Index size (KB): {(math.ceil(os.path.getsize('index_dir/terms.idx')) + math.ceil(os.path.getsize('index_dir/inverted_indx.bin'))) / 1024}\n")
        elapsed = end - start
        print(f"Elapsed: {elapsed:.6f} seconds")

    with open("docID_mapping.txt", "w") as f:
        for key, value in inverted_indx.doc_id_mapping.items():
            f.write(f"{key}: {value}\n")