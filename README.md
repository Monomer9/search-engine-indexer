# Search Engine: Indexing & Query Processing

This project was developed as part of a team-based system, with a focus on scalable indexing and ranking logic. It processes crawled web pages, builds an inverted index, and supports efficient ranked retrieval using TF-IDF and additional relevance signals.

## Overview

The system is divided into two main stages:

1. **Indexing Pipeline**
   - Parses HTML documents
   - Extracts and normalizes text
   - Builds an inverted index mapping terms to documents
   - Stores the index efficiently on disk

2. **Search Engine**
   - Processes user queries
   - Retrieves relevant documents using the inverted index
   - Ranks results using TF-IDF and importance-based scoring

This project was developed as part of a team, with a focus on building scalable indexing and ranking logic.

---

## My Contributions

I worked primarily on the indexing and search components. My contributions include:

- Implemented term importance scoring based on HTML structure (title, headings, bold text)
- Added bigram indexing and query support for improved phrase matching
- Built duplicate page detection using hashing to avoid redundant indexing
- Enhanced ranking using TF-IDF with additional weighting and normalization
- Added anchor text indexing to improve relevance
- Improved query processing and evaluation of search results

---

## Features

- Inverted index construction
- Disk-based index storage (binary format for efficiency)
- TF-IDF ranking with importance boosting
- Bigram (phrase) support
- Anchor text indexing
- Duplicate document detection
- Query normalization (tokenization + stemming)
- Efficient query processing using memory-mapped files

---

## Concepts Demonstrated

- Information Retrieval Systems
- Inverted Index Design
- TF-IDF Scoring
- Text Processing & Tokenization
- Disk-based Data Structures
- Caching and Performance Optimization
- HTML Parsing and Content Extraction

---

## How It Works

### Indexing
- HTML documents are parsed using BeautifulSoup
- Text is extracted and tokenized
- Tokens are stemmed and augmented with bigrams
- Terms are stored in an inverted index with:
  - Document frequency
  - Term frequency
  - Importance scores

- Partial indexes are written to disk and later merged into:
  - `terms.idx` (term → offset mapping)
  - `inverted_indx.bin` (postings lists)

### Search
- Queries are normalized (tokenized + stemmed + bigrams)
- Relevant postings are retrieved from disk
- Documents are intersected (AND queries)
- Results are ranked using:
  - TF-IDF
  - Importance boosting
  - Phrase bonus

---

## Project Structure

indexer.py
search.py
parse_index.py
evaluate.py

---

## How to Run

Note: The dataset of crawled HTML documents is not included due to size. The system expects JSON files containing page content and URLs.

### 1. Build the Index
python indexer.py

### 2. Run Search Engine
python search.py

### 3. Example Query
machine learning

## Example Output
Top 5 results:
https://example.com/page1
https://example.com/page2
...
Search took 12.34 ms.


---

## Technologies

- Python
- BeautifulSoup (HTML parsing)
- NLTK (stemming)
- mmap (efficient file access)
- JSON / Binary file handling

---

## Related Components

This project works with a web crawler that collects and stores documents for indexing:

- Web Crawler: [https://github.com/Monomer9/web-crawler.git]

---

## Requirements
- Python 3.x
- BeautifulSoup
- nltk

## What I Learned

This project gave me hands-on experience with how search engines process, store, and retrieve information at scale. I learned how indexing strategies, ranking algorithms, and performance optimizations impact the effectiveness and efficiency of search systems.

## Attribution

This project was originally developed as part of a team project. The original repository can be found here: [https://github.com/chrislnguye8/search-engine]

I contributed primarily to the indexing and search components as described above.
