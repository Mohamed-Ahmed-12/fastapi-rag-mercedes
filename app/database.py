import os
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase , sessionmaker
from dotenv import load_dotenv

from app.ai_models import get_embedding_model

load_dotenv()

class Base(DeclarativeBase):
    pass

        
# ====================== Database =====================
def create_db_engine():
    """
    Creates a SQLAlchemy engine for connecting to the PostgreSQL database.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")
    
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        raise

def create_db_session():
    
    """
    Creates a new SQLAlchemy session for interacting with the database.
    """
    engine = create_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_db():
    db = create_db_session()
    try:
        yield db
    finally:
        db.close()


def apply_schema():
    engine = create_db_engine()
    try:
        with engine.begin() as conn:
            # Enable pgvector extension before creating tables
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print("✅ pgvector extension is enabled.")

            # Create all tables defined in Base
            Base.metadata.create_all(conn)
            print("✅ Database schema applied successfully (Manuals & Chunks).")
            
    except Exception as e:
        print(f"❌ Error applying schema: {e}")
        raise 
    
# Vector store configuration
def get_vector_store():
    """
    Configures and returns a PGVector store using langchain_postgres.
    """
    try:
        CONNECTION_STRING = (
            f"postgresql+psycopg://postgres:admin@localhost:5432/mercedes_db"
        )

        return PGVector(
            connection=CONNECTION_STRING,
            collection_name="manual_chunks",
            embeddings=get_embedding_model(),
            use_jsonb=True,
        )
    except Exception as e:
        print(f"Error configuring vector store: {e}")
        raise