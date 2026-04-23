from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .database import Base

class MercedesManual(Base):
    __tablename__ = 'mercedes_manuals'
    
    id = Column(Integer, primary_key=True)
    chassis_code = Column(String(10), index=True, nullable=False) # E.g. "W205", "W213", "W222"
    year = Column(Integer, nullable=False) # Model year, e.g. 2015, 2020
    model = Column(String(50)) # E.g. "C-Class", "E-Class", "S-Class"
    language = Column(String(5), default="en") # ISO language code, e.g. "en", "de", "fr"
    file_name = Column(String(255)) # Good for tracking source files
    content_hash = Column(String(64), unique=True , nullable=False) # Hash of the PDF content for deduplication
    
    created_at = Column(String(50)) # Timestamp for when the manual was added to the DB
    updated_at = Column(String(50)) # Timestamp for when the manual was last updated in the DB
    
    title = Column(String(255)) # Extracted title from the PDF, e.g. "2011 C-Class Operator's Manual"
    slug = Column(String(255), unique=True) # URL-friendly identifier, e.g. "2011-c-class-operators-manual"
    description = Column(Text) # Optional description or summary of the manual
    source_url = Column(String(255)) # Original URL where the PDF was obtained, if applicable   
    
    # Prevents duplicate manuals for the same car/year/language
    __table_args__ = (
        UniqueConstraint('chassis_code', 'year', 'language', name='_manual_uc'),
    )

    chunks = relationship("ManualChunk", back_populates="manual", cascade="all, delete-orphan")

class ManualChunk(Base):
    __tablename__ = 'manual_chunks'
    
    id = Column(Integer, primary_key=True)
    manual_id = Column(Integer, ForeignKey('mercedes_manuals.id'), nullable=False)
    content = Column(Text, nullable=False)
    
    # Stores page_number, section_title, or original chunk index
    metadata_info = Column(JSON)
    
    # 768 is perfect for Llama-3 (Groq) or BERT
    embedding = Column(Vector(768)) 

    manual = relationship("MercedesManual", back_populates="chunks")

    # Add a vector index for performance
    __table_args__ = (
        Index(
            'hnsw_index_chunks',
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )