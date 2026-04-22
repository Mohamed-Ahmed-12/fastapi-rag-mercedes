"""
Indexing and Ingestion Script for RAG Application \n
Run this once to build the database. It handles downloading, splitting, and saving.
"""
"""
Mercedes-Benz C-Class Operator Manual - RAG Chunking Pipeline
Based on Databricks Ultimate Guide to Chunking Strategies
"""

import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import nltk

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from .config import get_embedding_model, DB_DIR



try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ─────────────────────────────────────────────
# 1. CONTENT TYPE DETECTION
# ─────────────────────────────────────────────

@dataclass
class ManualSection:
    """Represents a detected section of the manual."""
    content_type: str          # "procedure", "warning", "table", "spec", "concept", "diagram"
    section_path: List[str]    # ["Safety", "Air bags", "Front air bags"]
    raw_text: str
    has_warning: bool = False
    has_numbered_steps: bool = False
    has_table: bool = False
    system_tags: List[str] = field(default_factory=list)
    page_hint: Optional[int] = None


class ManualContentClassifier:
    """
    Detects content type for each section of the Mercedes manual.
    Drives which Databricks chunking strategy to apply.
    """

    # 1 Section header patterns (H1 → H3)
    HEADER_PATTERNS = [
        (1, re.compile(r"^#{1}\s+(.+)$", re.MULTILINE)),
        (2, re.compile(r"^#{2}\s+(.+)$", re.MULTILINE)),
        (3, re.compile(r"^#{3}\s+(.+)$", re.MULTILINE)),
        (3, re.compile(r"^\*\*(.+)\*\*$", re.MULTILINE)),  # bold titles in MD
    ]

    # 2 Warning block markers from the manual's own symbol system
    WARNING_PATTERNS = re.compile(
        r"(⚠\s*WARNING|Warning triangle icon|warning icon WARNING|"
        r"DANGER|CAUTION|Risk of accident|Risk of injury|Risk of explosion)",
        re.IGNORECASE,
    )

    # 3 Numbered/bulleted procedure steps
    STEP_PATTERNS = re.compile(
        r"(►\s.+|▶\s.+|\d+\.\s.+|\*\s.+|•\s.+)", re.MULTILINE
    )

    # 4 Table markers
    TABLE_PATTERNS = re.compile(r"(\|.+\|.+\||\+[-+]+\+)", re.MULTILINE)

    # 5 Display message blocks
    DISPLAY_MSG_PATTERNS = re.compile(
        r"(Display messages|Possible causes/consequences and ► Solutions|"
        r"Check .+ See Operator|Inoperative See|Currently Unavailable)",
        re.IGNORECASE,
    )

    # 6 Technical specification markers
    SPEC_PATTERNS = re.compile(
        r"(\d+\s*(mph|km/h|psi|kPa|bar|°F|°C|lbs|kg|in|mm|US qt|US gal|"
        r"Nm|lb-ft|V|A|W)\b|capacity|filling|tightening torque)",
        re.IGNORECASE,
    )

    # 7 Automotive system keywords for tagging
    SYSTEM_KEYWORDS = {
        "ABS": ["abs", "anti-lock braking"],
        "ESP": ["esp", "electronic stability"],
        "SRS": ["srs", "supplemental restraint", "air bag", "airbag", "etd"],
        "DISTRONIC": ["distronic", "adaptive cruise"],
        "PARKTRONIC": ["parktronic", "parking guidance"],
        "Climate": ["climate control", "a/c", "air conditioning", "coolant"],
        "Transmission": ["selector lever", "automatic transmission", "gear"],
        "Tires": ["tire", "tyre", "wheel", "tread", "pressure"],
        "Engine": ["engine oil", "coolant", "catalytic", "ignition"],
        "Safety": ["seat belt", "child seat", "restraint", "air bag"],
        "Lighting": ["headlamp", "fog lamp", "turn signal", "indicator lamp"],
    }

    def classify(self, text: str, section_path: List[str]) -> str:
        """Return content_type string for a text block."""
        text_lower = text.lower()

        has_warning = bool(self.WARNING_PATTERNS.search(text))
        has_steps = bool(self.STEP_PATTERNS.search(text))
        has_table = bool(self.TABLE_PATTERNS.search(text))
        has_display_msg = bool(self.DISPLAY_MSG_PATTERNS.search(text))
        has_specs = bool(self.SPEC_PATTERNS.search(text))

        # Priority order matches Databricks guide's "document structure" criterion
        if has_display_msg:
            return "display_message"
        if has_warning and has_steps:
            return "warning_procedure"  # Context-enriched chunking
        if has_steps and not has_warning:
            return "procedure"          # Recursive chunking
        if has_warning and not has_steps:
            return "warning_only"       # Context-enriched chunking
        if has_table and has_specs:
            return "spec_table"         # Fixed-size + metadata
        if has_table:
            return "reference_table"    # Fixed-size + metadata
        if has_specs:
            return "technical_spec"     # Adaptive chunking
        return "concept"               # Semantic / adaptive chunking

    def extract_system_tags(self, text: str) -> List[str]:
        """Tag which vehicle systems are referenced in the chunk."""
        tags = []
        text_lower = text.lower()
        for system, keywords in self.SYSTEM_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(system)
        return tags


# ─────────────────────────────────────────────
# 2. STRATEGY IMPLEMENTATIONS
# ─────────────────────────────────────────────

class RecursiveManualChunker:
    """
    Databricks Strategy #3 — Recursive Chunking
    Used for: numbered procedures, step-by-step instructions

    The manual uses ► and ▶ as step markers.
    We add these to the separator hierarchy so steps never split.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        # Separator hierarchy from the Databricks guide, adapted for manual syntax
        self.splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n\n",          # Major section breaks
                "\n\n",            # Paragraph breaks
                "\n►",             # New step (Mercedes uses ►)
                "\n▶",             # Alternate step marker
                "\n- ",            # Bullet points
                ". ",              # Sentence boundary
                " ",
                "",
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(
        self, text: str, metadata: dict
    ) -> List[Document]:
        raw_chunks = self.splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(raw_chunks):
            # Detect if this chunk contains a complete procedure unit
            step_count = len(re.findall(r"[►▶]\s", chunk))
            doc_meta = {
                **metadata,
                "chunk_id": f"{metadata.get('section_id','sec')}_{i:03d}",
                "chunk_type": "procedure",
                "strategy": "recursive",
                "step_count": step_count,
                "chunk_index": i,
                "total_chunks_in_section": len(raw_chunks),
            }
            documents.append(Document(page_content=chunk, metadata=doc_meta))

        return documents


class ContextEnrichedWarningChunker:
    """
    Databricks Strategy #5 — Context-Enriched Chunking
    Used for: WARNING blocks + their associated procedures

    Key insight from Databricks guide:
    "Helps maintain coherence across different parts of the document"
    → Perfect for warnings that must stay paired with their safety context.

    Implementation: window_size=1 attaches preceding section header
    and following procedure to each warning block.
    """

    def __init__(self, chunk_size: int = 500, window_size: int = 1):
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.base_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=50,
        )

    def chunk(
        self, text: str, metadata: dict
    ) -> List[Document]:
        # Split into base chunks first
        base_chunks = self.base_splitter.split_text(text)
        documents = []

        classifier = ManualContentClassifier()
        warning_pattern = ManualContentClassifier.WARNING_PATTERNS

        for i, chunk in enumerate(base_chunks):
            is_warning = bool(warning_pattern.search(chunk))

            # Build context window (preceding + following chunks)
            window_start = max(0, i - self.window_size)
            window_end = min(len(base_chunks), i + self.window_size + 1)

            preceding = " ".join(base_chunks[window_start:i])
            following = " ".join(base_chunks[i + 1 : window_end])

            if is_warning:
                # Enrich: prepend section context, append what to do
                enriched_text = (
                    f"[SECTION CONTEXT: {metadata.get('section_path','')[-1] if metadata.get('section_path') else ''}]\n"
                    f"[PRECEDING CONTEXT]: {preceding[:200]}\n\n"
                    f"[WARNING CONTENT]: {chunk}\n\n"
                    f"[FOLLOW-UP ACTION]: {following[:300]}"
                )
                chunk_subtype = "warning_enriched"
            else:
                enriched_text = chunk
                chunk_subtype = "procedure_step"

            doc_meta = {
                **metadata,
                "chunk_id": f"{metadata.get('section_id','sec')}_w{i:03d}",
                "chunk_type": chunk_subtype,
                "strategy": "context_enriched",
                "is_warning": is_warning,
                "has_preceding_context": bool(preceding),
                "has_following_context": bool(following),
                "chunk_index": i,
            }
            documents.append(
                Document(page_content=enriched_text, metadata=doc_meta)
            )

        return documents


class SemanticDisplayMessageChunker:
    """
    Databricks Strategy #2 — Semantic Chunking
    Used for: Display message tables (the largest repeating pattern in this manual)

    The manual has hundreds of problem → cause → solution blocks.
    Each block = one semantic unit. We detect and preserve these.
    """

    # Pattern: message header + problem + solutions
    BLOCK_PATTERN = re.compile(
        r"(Display messages.*?(?=Display messages|\Z))",
        re.DOTALL | re.IGNORECASE,
    )

    # Individual problem/solution pair
    PAIR_PATTERN = re.compile(
        r"(?P<problem>[A-Z][^\n]+(?:\n(?![►▶•]).*)*?)\n"
        r"(?P<solutions>(?:[►▶•].*\n?)+)",
        re.MULTILINE,
    )

    def chunk(
        self, text: str, metadata: dict
    ) -> List[Document]:
        documents = []

        # Try to extract structured problem/solution pairs first
        pairs = list(self.PAIR_PATTERN.finditer(text))

        if pairs:
            for i, match in enumerate(pairs):
                problem = match.group("problem").strip()
                solutions = match.group("solutions").strip()

                # Extract warning level from problem text
                warning_level = "info"
                if re.search(r"Risk of accident|Risk of injury", problem, re.IGNORECASE):
                    warning_level = "critical"
                elif re.search(r"inoperative|malfunction", problem, re.IGNORECASE):
                    warning_level = "warning"

                structured_text = (
                    f"DISPLAY MESSAGE — {metadata.get('section_path', [''])[0]}\n"
                    f"Condition: {problem}\n"
                    f"Actions:\n{solutions}"
                )

                doc_meta = {
                    **metadata,
                    "chunk_id": f"{metadata.get('section_id','sec')}_dm{i:03d}",
                    "chunk_type": "display_message",
                    "strategy": "semantic",
                    "warning_level": warning_level,
                    "semantic_density": round(
                        len(set(problem.lower().split()))
                        / max(1, len(problem.split())),
                        2,
                    ),
                }
                documents.append(
                    Document(page_content=structured_text, metadata=doc_meta)
                )
        else:
            # Fallback: sentence-boundary semantic split
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " "],
                chunk_size=500,
                chunk_overlap=80,
            )
            for i, chunk in enumerate(splitter.split_text(text)):
                doc_meta = {
                    **metadata,
                    "chunk_id": f"{metadata.get('section_id','sec')}_s{i:03d}",
                    "chunk_type": "display_message_fallback",
                    "strategy": "semantic_fallback",
                }
                documents.append(Document(page_content=chunk, metadata=doc_meta))

        return documents


class AdaptiveConceptChunker:
    """
    Databricks Strategy #4 — Adaptive Chunking
    Used for: System explanations (ABS, ESP, DISTRONIC PLUS, ATTENTION ASSIST...)

    The manual's concept sections vary greatly in complexity:
    - Simple: "ABS prevents wheels from locking" → large chunk OK
    - Complex: DISTRONIC PLUS sensor fusion details → needs smaller chunks

    Complexity metric: lexical density (unique/total word ratio)
    Maps to chunk_size inversely.
    """

    MIN_CHUNK = 300
    MAX_CHUNK = 900

    def __init__(self):
        self.stopwords = set(
            "the and is of to a in that it with as for at be this".split()
        )

    def _lexical_density(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        content = [w for w in words if w not in self.stopwords]
        return min(1.0, len(set(content)) / max(1, len(content)) / 0.8)

    def _sentence_complexity(self, text: str) -> float:
        """Long sentences = more complex."""
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = text.split(". ")
        if not sentences:
            return 0.0
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        return min(1.0, avg_len / 200)

    def _complexity_score(self, text: str) -> float:
        return (self._lexical_density(text) + self._sentence_complexity(text)) / 2

    def chunk(
        self, text: str, metadata: dict
    ) -> List[Document]:
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in text.split(". ") if s.strip()]

        documents = []
        current: List[str] = []
        current_size = 0
        current_complexity = 0.5

        def flush_chunk(idx: int) -> Document:
            joined = " ".join(current)
            complexity = self._complexity_score(joined)
            doc_meta = {
                **metadata,
                "chunk_id": f"{metadata.get('section_id','sec')}_a{idx:03d}",
                "chunk_type": "concept",
                "strategy": "adaptive",
                "text_complexity": round(complexity, 3),
                "chunk_size_chars": len(joined),
                "adaptive_target": int(
                    self.MAX_CHUNK
                    - complexity * (self.MAX_CHUNK - self.MIN_CHUNK)
                ),
            }
            return Document(page_content=joined, metadata=doc_meta)

        chunk_idx = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            sentence_complexity = self._complexity_score(sentence)

            if current:
                current_complexity = (current_complexity + sentence_complexity) / 2

            target = int(
                self.MAX_CHUNK
                - current_complexity * (self.MAX_CHUNK - self.MIN_CHUNK)
            )

            if current_size + sentence_len > target and current:
                documents.append(flush_chunk(chunk_idx))
                chunk_idx += 1
                # Overlap: keep last sentence for continuity
                current = [current[-1], sentence] if current else [sentence]
                current_size = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_size += sentence_len

        if current:
            documents.append(flush_chunk(chunk_idx))

        return documents


class FixedSizeSpecChunker:
    """
    Databricks Strategy #1 — Fixed-Size Chunking
    Used for: Technical spec tables, tire pressure tables, capacities

    Why fixed here? Spec tables are already uniform in structure.
    The Databricks guide notes: "Works well for content with consistent formatting."
    Tables must never be split mid-row.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 0):
        # No overlap for tables — rows are self-contained
        self.splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(
        self, text: str, metadata: dict
    ) -> List[Document]:
        # Detect and keep table header row with each chunk
        lines = text.split("\n")
        header_row = None
        for line in lines[:5]:  # Header is usually in first few lines
            if re.match(r"\|.+\|.+\|", line) or re.match(r"^\s*\w.*\|", line):
                header_row = line
                break

        raw_chunks = self.splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(raw_chunks):
            # Re-attach header to every chunk except the first
            final_content = chunk
            if i > 0 and header_row and header_row not in chunk:
                final_content = f"[TABLE HEADER]: {header_row}\n{chunk}"

            doc_meta = {
                **metadata,
                "chunk_id": f"{metadata.get('section_id','sec')}_sp{i:03d}",
                "chunk_type": "spec_table",
                "strategy": "fixed_size",
                "chunk_index": i,
                "total_chunks_in_section": len(raw_chunks),
                "has_table_header_reattached": (i > 0 and header_row is not None),
            }
            documents.append(
                Document(page_content=final_content, metadata=doc_meta)
            )

        return documents


# ─────────────────────────────────────────────
# 3. MAIN ORCHESTRATOR — Hybrid Chunking Pipeline
# ─────────────────────────────────────────────

class MercedesManualChunkingPipeline:
    """
    Hybrid chunking pipeline implementing the Databricks guide recommendation:
    "When in doubt, mix and match strategies to handle different content types."

    Routing logic:
      display_message     → SemanticDisplayMessageChunker
      warning_procedure   → ContextEnrichedWarningChunker
      warning_only        → ContextEnrichedWarningChunker
      procedure           → RecursiveManualChunker
      spec_table          → FixedSizeSpecChunker
      reference_table     → FixedSizeSpecChunker
      technical_spec      → AdaptiveConceptChunker
      concept             → AdaptiveConceptChunker
    """

    SECTION_HIERARCHY = [
        "Introduction", "At a glance", "Safety", "Opening/closing",
        "Seats, steering wheel and mirrors", "Lights and windshield wipers",
        "Climate control", "Driving and parking",
        "On-board computer and displays", "Stowing and features",
        "Maintenance and care", "Breakdown assistance",
        "Wheels and tires", "Technical data",
    ]

    def __init__(self):
        self.classifier = ManualContentClassifier()
        self.chunkers = {
            "display_message":   SemanticDisplayMessageChunker(),
            "warning_procedure": ContextEnrichedWarningChunker(chunk_size=500),
            "warning_only":      ContextEnrichedWarningChunker(chunk_size=400),
            "procedure":         RecursiveManualChunker(chunk_size=600, chunk_overlap=60),
            "spec_table":        FixedSizeSpecChunker(chunk_size=800),
            "reference_table":   FixedSizeSpecChunker(chunk_size=600),
            "technical_spec":    AdaptiveConceptChunker(),
            "concept":           AdaptiveConceptChunker(),
        }

    def _parse_sections(self, full_text: str) -> List[Tuple[List[str], str]]:
        """
        Parse the manual into (section_path, text) pairs using
        header detection. Returns ordered list.
        """
        section_pattern = re.compile(
            r"^(#{1,3})\s+(.+)$", re.MULTILINE
        )

        sections = []
        matches = list(section_pattern.finditer(full_text))

        breadcrumb = []

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
            body = full_text[start:end].strip()

            # Maintain breadcrumb trail
            while len(breadcrumb) >= level:
                breadcrumb.pop()
            breadcrumb.append(title)

            if body:
                sections.append((list(breadcrumb), body))

        return sections

    def _build_section_id(self, section_path: List[str]) -> str:
        """Create a slug ID from section path."""
        return "_".join(
            re.sub(r"[^\w]", "", p.lower().replace(" ", "_"))[:20]
            for p in section_path
        )

    def process(self, full_manual_text: str, source_url: str = "") -> List[Document]:
        """
        Full pipeline: parse → classify → chunk → enrich metadata.
        Returns list of Documents ready for embedding.
        """
        all_documents: List[Document] = []
        sections = self._parse_sections(full_manual_text)

        print(f"[Pipeline] Parsed {len(sections)} sections from manual")

        for section_path, section_text in sections:
            content_type = self.classifier.classify(section_text, section_path)
            system_tags = self.classifier.extract_system_tags(section_text)
            section_id = self._build_section_id(section_path)

            # ── Base metadata attached to every chunk (Databricks best practice #4)
            base_metadata = {
                "source": "Mercedes-Benz C-Class Operator Manual W204",
                "source_url": source_url,
                "section_path": " > ".join(section_path),
                "section_path_list": section_path,
                "section_id": section_id,
                "top_level_section": section_path[0] if section_path else "Unknown",
                "content_type": content_type,
                "system_tags": system_tags,
                "has_warning": content_type in ("warning_procedure", "warning_only"),
                "is_safety_critical": any(
                    s in ("Safety", "Driving and parking", "Breakdown assistance")
                    for s in section_path
                ),
            }

            # ── Route to correct chunker
            chunker = self.chunkers.get(content_type, self.chunkers["concept"])
            chunks = chunker.chunk(section_text, base_metadata)
            all_documents.extend(chunks)

        print(f"[Pipeline] Total chunks produced: {len(all_documents)}")
        return all_documents


def run_indexing_pipeline():
    # 1. Check if DB already exists to save time/resources
    if os.path.exists(DB_DIR):
        print(f"Directory '{DB_DIR}' already exists. Skipping indexing.")
        return

    print("Starting indexing process...")
    
    # 2. Load PDF
    file_path = "https://www.mbusa.com/css-oom/assets/en-us/pdf/mercedes-c-class-sedan-2011-w204-operators-manual-1.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Chunking Pipeline (can be moved to a separate module for better organization)
    pipeline = MercedesManualChunkingPipeline()

    print("Running chunking pipeline...")
    
    chunks = pipeline.process(
        docs,
        source_url="../data/mercedes_c_class_manual.md",
    )

    # # 3. Split Text (Chunking)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, 
    #     chunk_overlap=200, 
    #     add_start_index=True
    # )
    
    # # Chunks
    # all_splits = text_splitter.split_documents(docs)

    # 4. Create Embeddings
    embeddings = get_embedding_model()
    
    # 5. Create and Persist Vector Store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Successfully indexed {len(chunks)} chunks into {DB_DIR}")

