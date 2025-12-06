"""
etl_pipeline.py

ETL pipeline for extracting:
 - Citations (e.g., "Federal Decree-Law No. (47) of 2022 ...")
 - Term / definition pairs (terms and their definitions)

Approach:
1. Deterministic pass (regex + rule-based normalization)
2. Non-deterministic pass (semantic similarity using sentence-transformer if available,
   otherwise TF-IDF + cosine similarity) to find candidates missed or ambiguous in deterministic pass.

Outputs: JSON (structured canonical IDs, source doc, page, original text, normalized id, term, definition)
Optional: upload outputs to AWS S3 (minimal demonstration using boto3)
"""
import os
import re
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# PDF reading
try:
    import pdfplumber
except Exception:
    pdfplumber = None
    from PyPDF2 import PdfReader

# ML & text processing
try:
    # preferred: sentence-transformers if available for semantic matching
    from sentence_transformers import SentenceTransformer, util as st_util
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMER_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ_AVAILABLE = True
except Exception:
    _RAPIDFUZZ_AVAILABLE = False

# AWS (minimal usage demonstration)
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False

# -------------------------
# Logging setup (minimal)
# -------------------------
logger = logging.getLogger("DocumentsETL")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

# -------------------------
# Helper dataclasses
# -------------------------
@dataclass
class ExtractionRecord:
    source_file: str
    page_num: int
    raw_text: str
    type: str  # 'citation' or 'term_definition'
    term: str = ""
    definition: str = ""
    canonical_id: str = ""  # e.g., fed_decree_law_47_2022
    confidence: float = 1.0  # 0..1, higher = more confident

# -------------------------
# Canonicalization utilities
# -------------------------
def canonicalize_title_to_id(title: str) -> str:
    """
    Example:
      "Federal Decree by Law No. (47) of 2022 Concerning Corporate and Business Tax"
    -> "fed_decree_law_47_2022"
    Approach: lower, remove stopwords, extract numbers, keep important short tokens.
    """
    if not title:
        return ""
    text = title.lower()
    # common normalization
    text = re.sub(r"[^a-z0-9\(\)\s]", " ", text)
    # extract numbers in parentheses or "No. 47" patterns
    nums = re.findall(r"\bno\.?\s*\(?(\d{1,4})\)?|\((\d{1,4})\)|\b(\d{4})\b", title, flags=re.I)
    # flatten tuple matches
    nums_flat = [n for tpl in nums for n in tpl if n]
    # pick meaningful tokens
    tokens = re.findall(r"\b(federal|decree|law|cabinet|resolution|act|regulation|ministerial|order|amendment|no|number|of|the|concerning|concerning|concerning|regarding|on|concerning|concerning|concerning)\b|\b[a-z]{2,}\b", text)
    # crude token filter keep short core tokens
    keep = []
    for t in re.findall(r"[a-z0-9]+", text):
        if t in {"the","of","and","by","no","concerning","regarding","on","in","for","a","an"}:
            continue
        if len(t) <= 1:
            continue
        keep.append(t)
    # prefer core type tokens (federal, decree, law, cabinet, resolution)
    important = [t for t in keep if t in ("federal", "decree", "law", "cabinet", "resolution", "act", "regulation")]
    if not important and keep:
        important = keep[:2]
    # build id
    parts = []
    if important:
        parts.extend(important[:3])
    # append numbers
    if nums_flat:
        parts.extend(nums_flat[:2])
    # fallback: use first three words
    if not parts:
        parts = keep[:3]
    id_str = "_".join(parts)
    id_str = re.sub(r"_+", "_", id_str).strip("_")
    return id_str

def canonicalize_term(term: str) -> str:
    if not term:
        return ""
    t = term.strip().lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", "_", t).strip("_")
    return t

# -------------------------
# Main ETL class
# -------------------------
class DocumentsETL:
    def __init__(self, input_dir: str, debug: bool = False, ml_threshold: float = 0.65):
        self.input_dir = input_dir
        self.debug = debug
        self.ml_threshold = ml_threshold
        self.records: List[ExtractionRecord] = []
        # initialize ML model if available
        self.sentence_model = None
        if _SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                logger.info("Loading sentence-transformer model (for semantic matching).")
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning("SentenceTransformer load failed, falling back to TF-IDF. Error: %s", e)
                self.sentence_model = None

    # -------------------------
    # PDF text extraction
    # -------------------------
    def extract_text_from_pdf(self, filepath: str) -> List[Tuple[int, str]]:
        """
        Return list of tuples (page_num (1-indexed), page_text)
        Uses pdfplumber if present else PyPDF2 as fallback.
        """
        pages = []
        logger.info("Extracting text from %s", filepath)
        try:
            if pdfplumber:
                with pdfplumber.open(filepath) as pdf:
                    for i, p in enumerate(pdf.pages):
                        text = p.extract_text() or ""
                        pages.append((i + 1, text))
            else:
                reader = PdfReader(filepath)
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    pages.append((i + 1, text))
            return pages
        except Exception as e:
            logger.exception("Failed to read PDF %s: %s", filepath, e)
            return []

    # -------------------------
    # Deterministic extraction (regex/rule-based)
    # -------------------------
    CITATION_REGEXES = [
        # common patterns for "Federal Decree-Law No. (47) of 2022 ..." etc
        re.compile(r"\b(Federal\s+Decree[-\s]?Law|Federal\s+Decree\s+by\s+Law|Federal\s+Decree|Cabinet\s+Resolution|Cabinet\s+Resolution\s+No\.|Cabinet\s+Resolution\s+No)\b[^\n,.]{0,120}\bNo\.?\s*\(?\d{1,4}\)?[^\n]{0,80}of\s+\d{4}", flags=re.I),
        re.compile(r"\b(Cabinet\s+Resolution|Cabinet\s+Resolution\s+No\.?|Federal\s+Decree[-\s]?Law|Ministerial\s+Decree)[^\n]{0,140}", flags=re.I),
        # generic "No. (123) of 2021" capture
        re.compile(r"\bNo\.?\s*\(?\d{1,4}\)?\s*of\s*\d{4}\b", flags=re.I),
    ]

    TERM_DEF_PATTERNS = [
        # pattern: "Term: <TERM>\nDefinition: <DEFINITION>"
        re.compile(r"Term\s*[:\-]\s*(?P<term>[A-Z][A-Za-z0-9\s\-\(\)]+?)\s*(?:Definition|Means)\s*[:\-]\s*(?P<def>.+?)(?=\n[A-Z][a-z]|$|\n\n|\nTerm\s*:)", flags=re.I | re.S),
        # pattern: "<Term> : <Definition>" single-line or multi-line
        re.compile(r"(?P<term>[A-Z][A-Za-z0-9\s]{2,80})\s*:\s*(?P<def>.+?)(?=\n[A-Z][a-z]|\n\n|$)", flags=re.S),
        # pattern: "Term — Definition" em-dash
        re.compile(r"(?P<term>[A-Z][A-Za-z0-9\s]{2,80})\s+[—\-–]\s+(?P<def>.+?)(?=\n[A-Z][a-z]|\n\n|$)", flags=re.S),
    ]

    def deterministic_extract(self, pages: List[Tuple[int, str]], source_file: str):
        """
        Run regex passes over text to find explicit citations and term/definition pairs.
        """
        for page_num, text in pages:
            if not text or text.strip() == "":
                continue
            # CITATIONS
            for cre in self.CITATION_REGEXES:
                for m in cre.finditer(text):
                    snippet = m.group(0).strip()
                    if len(snippet) > 10:
                        can_id = canonicalize_title_to_id(snippet)
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=snippet,
                            type="citation",
                            term="",
                            definition="",
                            canonical_id=can_id,
                            confidence=0.95,
                        )
                        self.records.append(rec)
                        if self.debug:
                            logger.info("Deterministic citation found: %s", snippet)

            # TERMS & DEFINITIONS
            for tre in self.TERM_DEF_PATTERNS:
                for m in tre.finditer(text):
                    term = (m.group("term") or "").strip()
                    definition = (m.group("def") or "").strip()
                    if term and definition:
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=(term + " : " + definition)[:1000],
                            type="term_definition",
                            term=term,
                            definition=definition,
                            canonical_id=canonicalize_term(term),
                            confidence=0.98,
                        )
                        self.records.append(rec)
                        if self.debug:
                            logger.info("Deterministic term/def found: %s => %s", term, (definition[:60] + "...") if len(definition) > 60 else definition)

    # -------------------------
    # Non-deterministic pass (semantic / ML) to improve recall
    # -------------------------
    def nondeterministic_refinement(self, pages: List[Tuple[int, str]], source_file: str):
        """
        Use ML/semantic similarity to find candidate sentences that look like definitions
        or citations but were missed by strict regex. Two strategies:
         - If sentence-transformer is available: encode sentences and compare to known patterns
         - Otherwise: TF-IDF + cosine similarity against high-confidence deterministic examples
        """
        # build list of high-confidence deterministic examples from current records for this file
        high_conf_citations = [r.raw_text for r in self.records if r.source_file == os.path.basename(source_file) and r.type == "citation" and r.confidence >= 0.9]
        high_conf_terms = [r.definition for r in self.records if r.source_file == os.path.basename(source_file) and r.type == "term_definition" and r.confidence >= 0.9]

        # tokenise pages into sentences (simple split)
        all_sentences = []
        sentence_locations = []  # (page_num, sentence)
        for page_num, text in pages:
            if not text:
                continue
            # naive split by newline and dot; we keep reasonably long sentences
            chunks = re.split(r"\n+", text)
            for chunk in chunks:
                sents = re.split(r"(?<=[\.\;\:])\s+", chunk)
                for s in sents:
                    s_clean = s.strip()
                    if len(s_clean) < 40:
                        continue
                    all_sentences.append(s_clean)
                    sentence_locations.append((page_num, s_clean))

        if not all_sentences:
            return

        # If we have deterministic citations/defs to compare against, use them as seeds
        seeds = high_conf_citations + high_conf_terms
        if not seeds:
            # if no seeds found (empty deterministic pass), create a few heuristic seeds for "definition-like" or "No. (xx) of yyyy"
            seeds = [
                "Federal Decree-Law No. (47) of 2022 Concerning Corporate and Business Tax",
                "Term: The period specified in Article 57 of this Decree-Law",
                "No. (7) of 2017 on Excise Tax"
            ]

        logger.info("Non-deterministic pass: %d sentences found, %d seed examples", len(all_sentences), len(seeds))

        if self.sentence_model:
            # encode seeds and sentences
            seed_emb = self.sentence_model.encode(seeds, convert_to_tensor=True)
            sent_emb = self.sentence_model.encode(all_sentences, convert_to_tensor=True)
            # compute similarity matrix (seeds x sentences)
            sim_matrix = st_util.pytorch_cos_sim(seed_emb, sent_emb).cpu().numpy()
            # for each sentence, consider best seed similarity
            import numpy as np
            best_sim = sim_matrix.max(axis=0)  # best seed similarity for each sentence
            for idx, score in enumerate(best_sim):
                if score >= self.ml_threshold:
                    page_num, sentence = sentence_locations[idx]
                    # decide whether it looks more like a citation or a definition
                    if re.search(r"\bNo\.?\s*\(?\d{1,4}\)?\s*of\s*\d{4}\b", sentence, flags=re.I) or re.search(r"Federal\s+Decree|Cabinet\s+Resolution|Decree", sentence, flags=re.I):
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=sentence,
                            type="citation",
                            canonical_id=canonicalize_title_to_id(sentence),
                            confidence=float(score)
                        )
                    else:
                        # guess term/definition split: split at ":" or "means" or "shall mean"
                        term, definition = self._guess_term_def_from_sentence(sentence)
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=sentence[:1000],
                            type="term_definition",
                            term=term,
                            definition=definition,
                            canonical_id=canonicalize_term(term),
                            confidence=float(score)
                        )
                    # avoid duplicates: check if similar raw_text exists already
                    if not self._is_duplicate(rec):
                        self.records.append(rec)
            return

        # fallback TF-IDF approach
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
            vsent = vectorizer.fit_transform(seeds + all_sentences)
            seed_vecs = vsent[:len(seeds)]
            sent_vecs = vsent[len(seeds):]
            sim = cosine_similarity(sent_vecs, seed_vecs)  # sentences x seeds
            best = sim.max(axis=1)
            for idx, score in enumerate(best):
                if score >= self.ml_threshold:
                    page_num, sentence = sentence_locations[idx]
                    if re.search(r"\bNo\.?\s*\(?\d{1,4}\)?\s*of\s*\d{4}\b", sentence, flags=re.I) or re.search(r"Federal\s+Decree|Cabinet\s+Resolution|Decree", sentence, flags=re.I):
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=sentence,
                            type="citation",
                            canonical_id=canonicalize_title_to_id(sentence),
                            confidence=float(score)
                        )
                    else:
                        term, definition = self._guess_term_def_from_sentence(sentence)
                        rec = ExtractionRecord(
                            source_file=os.path.basename(source_file),
                            page_num=page_num,
                            raw_text=sentence[:1000],
                            type="term_definition",
                            term=term,
                            definition=definition,
                            canonical_id=canonicalize_term(term),
                            confidence=float(score)
                        )
                    if not self._is_duplicate(rec):
                        self.records.append(rec)
        except Exception as e:
            logger.exception("TF-IDF fallback failure: %s", e)

    def _is_duplicate(self, new_rec: ExtractionRecord) -> bool:
        """
        Basic duplicate heuristic: same type + very similar raw_text or same canonical_id already in records.
        """
        for r in self.records:
            if r.type == new_rec.type and r.canonical_id and new_rec.canonical_id and r.canonical_id == new_rec.canonical_id:
                return True
            # fuzzy text compare
            if _RAPIDFUZZ_AVAILABLE:
                if fuzz.partial_ratio(r.raw_text[:200], new_rec.raw_text[:200]) > 90:
                    return True
            else:
                if new_rec.raw_text[:120] in r.raw_text[:120] or r.raw_text[:120] in new_rec.raw_text[:120]:
                    return True
        return False

    def _guess_term_def_from_sentence(self, sentence: str) -> Tuple[str, str]:
        """
        Heuristic: split on 'means', 'shall mean', ':' or 'is:' etc. Return (term, def).
        """
        s = sentence.strip()
        # candidates
        splitters = [r"\bshall mean\b", r"\bmeans\b", r"\bmean\b", r":", r"–", r"—", r"-"]
        for sp in splitters:
            parts = re.split(sp, s, maxsplit=1, flags=re.I)
            if len(parts) == 2:
                left = parts[0].strip(" .;,:\n\"'")
                right = parts[1].strip(" .;,\n\"'")
                # choose left as term if short-ish
                if 2 <= len(left.split()) <= 6:
                    return (left, right)
        # fallback: produce an artificial term (first 4 words) and the rest as definition
        words = s.split()
        term = " ".join(words[:4])
        definition = " ".join(words[4:]) if len(words) > 4 else ""
        return (term, definition)

    # -------------------------
    # Main process for one file
    # -------------------------
    def process_file(self, filepath: str):
        t0 = time.time()
        pages = self.extract_text_from_pdf(filepath)
        if not pages:
            logger.warning("No pages extracted for %s", filepath)
            return
        # deterministic first
        self.deterministic_extract(pages, filepath)
        # then non-deterministic refinement
        self.nondeterministic_refinement(pages, filepath)
        t1 = time.time()
        logger.info("Processed %s in %.2f sec; records so far: %d", os.path.basename(filepath), t1 - t0, len(self.records))

    # -------------------------
    # Save/Export functions
    # -------------------------
    def export_json(self, outpath: str):
        out_data = [asdict(r) for r in self.records]
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d records to JSON %s", len(self.records), outpath)

    def export_csv(self, outpath: str):
        import csv
        keys = ["source_file","page_num","type","term","definition","canonical_id","raw_text","confidence"]
        with open(outpath, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                row = {k: getattr(r, k) for k in keys}
                writer.writerow(row)
        logger.info("Exported %d records to CSV %s", len(self.records), outpath)

    # -------------------------
    # Minimal AWS S3 demo
    # -------------------------
    def upload_to_s3(self, local_path: str, bucket: str, key: str) -> bool:
        if not _BOTO3_AVAILABLE:
            logger.error("boto3 not installed; cannot upload to S3.")
            return False
        try:
            s3 = boto3.client("s3")
            s3.upload_file(local_path, bucket, key)
            logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, key)
            return True
        except (BotoCoreError, ClientError) as e:
            logger.exception("S3 upload failed: %s", e)
            return False

# -------------------------
# CLI / runner
# -------------------------
def find_pdfs(input_dir: str) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                files.append(os.path.join(root, fn))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Documents ETL Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="./output.json", help="Output JSON path")
    parser.add_argument("--output_csv", type=str, default="./output.csv", help="Output CSV path")
    parser.add_argument("--upload_s3", type=lambda x: x.lower() in ("true","1","yes"), default=False, help="Upload outputs to S3? (True/False)")
    parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket name (if upload_s3=True)")
    parser.add_argument("--s3_key_prefix", type=str, default="etl_outputs/", help="Prefix for S3 object keys")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--ml_threshold", type=float, default=0.65, help="Semantic similarity threshold (0..1)")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    start = time.time()

    pdfs = find_pdfs(args.input_dir)
    if not pdfs:
        logger.error("No PDFs found in %s", args.input_dir)
        return

    etl = DocumentsETL(args.input_dir, debug=args.debug, ml_threshold=args.ml_threshold)
    for p in pdfs:
        try:
            etl.process_file(p)
        except Exception as e:
            logger.exception("Processing failed for %s: %s", p, e)

    # export
    etl.export_json(args.output)
    etl.export_csv(args.output_csv)

    # optional S3 upload demo
    if args.upload_s3:
        if not args.s3_bucket:
            logger.error("S3 upload requested but no bucket provided.")
        else:
            # upload both outputs
            etl.upload_to_s3(args.output, args.s3_bucket, args.s3_key_prefix + os.path.basename(args.output))
            etl.upload_to_s3(args.output_csv, args.s3_bucket, args.s3_key_prefix + os.path.basename(args.output_csv))

    elapsed = time.time() - start
    logger.info("ETL complete in %.2f seconds. Total records: %d", elapsed, len(etl.records))

if __name__ == "__main__":
    main()
