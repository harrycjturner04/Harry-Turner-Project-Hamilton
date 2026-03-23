#!/usr/bin/env python3
"""
Database Builder for Chromatography/Mass Spectrometry Data

This module provides functionality to:
1. Automatically discover and process data files from a source directory
2. Distinguish between metadata and raw data files
3. Save raw data in efficient Parquet format
4. Build a vectorized metadata database for semantic search
5. Handle flexible data structures without hardcoded filenames

Author: Harry Turner Capstone Project
Date: December 2025
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CORE CONFIGURATION
# ============================================================================

METADATA_KEYWORDS = [
    'metadata', 'meta', 'param', 'parameters', 'config', 'settings'
]

RAW_DATA_KEYWORDS = [
    'combined', 'raw', 'data', 'measurements', 'results'
]


# ============================================================================
# FILE DISCOVERY AND CLASSIFICATION
# ============================================================================

def find_data_files(data_dir: Path | str) -> List[Path]:
    """
    Discover all data files in the specified directory.
    
    Args:
        data_dir: Path to directory containing data files
        
    Returns:
        List of Path objects for discovered files
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no data files found
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Supported file extensions
    extensions = ['*.csv', '*.xlsx', '*.xls', '*.parquet', '*.tsv']
    
    files = []
    for ext in extensions:
        files.extend(data_path.glob(ext))
    
    if not files:
        raise ValueError(f"No data files found in {data_path}")
    
    logger.info(f"Discovered {len(files)} data files in {data_path}")
    for file in files:
        logger.info(f"  - {file.name}")
    
    return sorted(files)


def classify_file_type(file_path: Path) -> str:
    """
    Classify a file as 'metadata' or 'raw_data' based on filename and content.
    
    Args:
        file_path: Path to the file to classify
        
    Returns:
        'metadata' or 'raw_data'
    """
    filename_lower = file_path.name.lower()
    
    # Check filename for metadata keywords
    for keyword in METADATA_KEYWORDS:
        if keyword in filename_lower:
            logger.info(f"Classified '{file_path.name}' as METADATA (filename match: '{keyword}')")
            return 'metadata'
    
    # Check for raw data keywords
    for keyword in RAW_DATA_KEYWORDS:
        if keyword in filename_lower:
            logger.info(f"Classified '{file_path.name}' as RAW DATA (filename match: '{keyword}')")
            return 'raw_data'
    
    # If no clear classification, inspect content
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=5)
        else:
            # Default to raw data for other formats
            logger.info(f"Classified '{file_path.name}' as RAW DATA (default)")
            return 'raw_data'
        
        # Heuristic: metadata tends to have more string/object columns
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        
        if numeric_ratio < 0.3:  # Less than 30% numeric columns
            logger.info(f"Classified '{file_path.name}' as METADATA (column analysis: {numeric_ratio:.1%} numeric)")
            return 'metadata'
        else:
            logger.info(f"Classified '{file_path.name}' as RAW DATA (column analysis: {numeric_ratio:.1%} numeric)")
            return 'raw_data'
            
    except Exception as e:
        logger.warning(f"Could not inspect '{file_path.name}': {e}. Defaulting to RAW DATA")
        return 'raw_data'


def split_metadata_and_raw(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Split files into metadata and raw data categories.
    
    Args:
        files: List of file paths to classify
        
    Returns:
        Tuple of (metadata_files, raw_data_files)
    """
    metadata_files = []
    raw_data_files = []
    
    for file_path in files:
        file_type = classify_file_type(file_path)
        if file_type == 'metadata':
            metadata_files.append(file_path)
        else:
            raw_data_files.append(file_path)
    
    logger.info(f"\nClassification summary:")
    logger.info(f"  Metadata files: {len(metadata_files)}")
    logger.info(f"  Raw data files: {len(raw_data_files)}")
    
    return metadata_files, raw_data_files


# ============================================================================
# RAW DATA PROCESSING
# ============================================================================

def sanitize_table_name(filename: str) -> str:
    """
    Convert a filename into a filesystem-safe table name.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized table name
    """
    # Remove extension
    name = Path(filename).stem
    
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w\s-]', '_', name)
    name = re.sub(r'[-\s]+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Convert to lowercase
    name = name.lower()
    
    return name


def load_dataframe(file_path: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load a data file into a pandas DataFrame.
    
    Args:
        file_path: Path to the file
        chunk_size: Optional chunk size for large files
        
    Returns:
        DataFrame containing the data
    """
    logger.info(f"Loading {file_path.name}...")
    
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Drop unnamed index columns that can cause issues with parquet conversion
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        raise


def load_and_save_raw_to_parquet(
    raw_files: List[Path],
    output_dir: Path,
    compression: str = 'snappy'
) -> Dict[str, Path]:
    """
    Load raw data files and save them as Parquet files.
    
    Args:
        raw_files: List of raw data file paths
        output_dir: Directory to save Parquet files
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', etc.)
        
    Returns:
        Dictionary mapping table names to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for file_path in raw_files:
        try:
            # Load the data
            df = load_dataframe(file_path)
            
            # Generate table name
            table_name = sanitize_table_name(file_path.name)
            output_path = output_dir / f"{table_name}.parquet"
            
            # Save as Parquet
            logger.info(f"Saving to {output_path.name}...")
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=compression,
                index=False
            )
            
            saved_files[table_name] = output_path
            logger.info(f"✓ Saved {table_name} → {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            continue
    
    return saved_files


# ============================================================================
# METADATA VECTORIZATION
# ============================================================================

# Global model instance (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model (singleton pattern).
    
    Returns:
        Initialized SentenceTransformer model
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Loading embedding model 'all-MiniLM-L6-v2'...")
        # This model is lightweight (80MB) and provides good quality embeddings
        # Embedding dimension: 384
        # Can be replaced with larger models for better quality:
        # - 'all-mpnet-base-v2' (420MB, dim=768) - higher quality
        # - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual support)
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Embedding model loaded successfully")
    
    return _embedding_model

def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Generate embeddings for text data using sentence-transformers.
    
    This implementation uses the 'all-MiniLM-L6-v2' model which provides:
    - High quality semantic embeddings
    - Fast inference (optimized for CPU and GPU)
    - Embedding dimension: 384
    - Trained on 1B+ sentence pairs
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding (larger = faster but more memory)
        show_progress: Whether to show progress bar
        
    Returns:
        Numpy array of shape (n_texts, 384)
    """
    logger.info(f"Generating embeddings for {len(texts)} text entries...")
    
    # Get the model
    model = get_embedding_model()
    
    # Generate embeddings
    # The model automatically handles:
    # - Tokenization
    # - Batching
    # - GPU acceleration (if available)
    # - Normalization
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
    return embeddings


def create_metadata_text_representation(row: pd.Series) -> str:
    """
    Create a text representation of a metadata row for embedding.
    
    Args:
        row: Pandas Series representing a metadata row
        
    Returns:
        String representation combining key information
    """
    text_parts = []
    
    for col, value in row.items():
        if pd.notna(value) and value != '':
            # Convert to string and clean
            val_str = str(value).strip()
            if val_str:
                text_parts.append(f"{col}: {val_str}")
    
    return " | ".join(text_parts)


def build_vectorised_metadata_db(
    metadata_files: List[Path],
    output_dir: Path
) -> Path:
    """
    Build a vectorized metadata database from metadata files.
    
    Args:
        metadata_files: List of metadata file paths
        output_dir: Directory to save the vectorized database
        
    Returns:
        Path to the saved vector database
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("BUILDING VECTORIZED METADATA DATABASE")
    logger.info("="*70)
    
    # Load and combine all metadata
    all_metadata = []
    
    for file_path in metadata_files:
        try:
            df = load_dataframe(file_path)
            df['source_file'] = file_path.name
            all_metadata.append(df)
        except Exception as e:
            logger.error(f"Failed to load metadata from {file_path.name}: {e}")
            continue
    
    if not all_metadata:
        logger.warning("No metadata files were successfully loaded")
        return None
    
    # Combine all metadata
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Clean up combined metadata: handle mixed types and remove problematic columns
    # Drop any remaining unnamed index columns
    combined_metadata = combined_metadata.loc[:, ~combined_metadata.columns.str.contains('^Unnamed')]
    
    # Convert all columns to string for safety (avoids mixed type issues)
    # This ensures everything can be serialized properly
    for col in combined_metadata.columns:
        if combined_metadata[col].dtype == 'object':
            try:
                combined_metadata[col] = combined_metadata[col].astype(str)
            except Exception:
                pass  # If conversion fails, keep original
    
    logger.info(f"Combined metadata: {len(combined_metadata):,} rows")
    
    # Create text representations
    logger.info("Creating text representations...")
    combined_metadata['text_representation'] = combined_metadata.apply(
        create_metadata_text_representation, axis=1
    )
    
    # Generate embeddings
    texts = combined_metadata['text_representation'].tolist()
    embeddings = embed_texts(texts)
    
    # Create vector database structure
    vector_db = {
        'id': list(range(len(combined_metadata))),
        'text': texts,
        'embedding': [emb.tolist() for emb in embeddings],
    }
    
    # Add original metadata columns
    for col in combined_metadata.columns:
        if col != 'text_representation':
            vector_db[col] = combined_metadata[col].tolist()
    
    # Save as Parquet (efficient for large data)
    vector_db_path = output_dir / 'vector_db.parquet'
    vector_df = pd.DataFrame({
        'id': vector_db['id'],
        'text': vector_db['text'],
        'embedding': vector_db['embedding'],
    })
    
    # Save embeddings separately as numpy array for efficient similarity search
    embeddings_path = output_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    
    # Save metadata - convert to safe types for parquet
    metadata_db_path = output_dir / 'metadata.parquet'
    metadata_for_save = combined_metadata.copy()
    
    # Convert all columns to string to avoid mixed type issues
    for col in metadata_for_save.columns:
        metadata_for_save[col] = metadata_for_save[col].astype(str)
    
    metadata_for_save.to_parquet(metadata_db_path, index=False)
    
    # Save index mapping
    index_path = output_dir / 'index.parquet'
    index_for_save = vector_df.copy()
    # Convert embedding to string for parquet compatibility
    index_for_save['embedding'] = index_for_save['embedding'].apply(
        lambda x: str(x) if not isinstance(x, str) else x
    )
    index_for_save.to_parquet(index_path, index=False)
    
    logger.info(f"✓ Saved vector database to {output_dir}")
    logger.info(f"  - Embeddings: {embeddings_path.name}")
    logger.info(f"  - Metadata: {metadata_db_path.name}")
    logger.info(f"  - Index: {index_path.name}")
    
    return output_dir


def similarity_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 10
) -> np.ndarray:
    """
    Perform similarity search using cosine similarity.
    
    Args:
        query_embedding: Query vector of shape (embedding_dim,)
        embeddings: Database embeddings of shape (n_docs, embedding_dim)
        top_k: Number of top results to return
        
    Returns:
        Indices of top-k most similar documents
    """
    # Compute cosine similarity
    similarities = np.dot(embeddings, query_embedding)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return top_indices


# ============================================================================
# HELPER FUNCTIONS FOR VECTOR DB USAGE
# ============================================================================

def load_vector_db(db_dir: Path | str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load a previously built vector database.
    
    Args:
        db_dir: Path to the vector database directory
        
    Returns:
        Tuple of (embeddings array, metadata DataFrame)
    """
    db_path = Path(db_dir)
    
    embeddings = np.load(db_path / 'embeddings.npy')
    metadata = pd.read_parquet(db_path / 'metadata.parquet')
    
    logger.info(f"Loaded vector DB: {len(embeddings):,} entries")
    return embeddings, metadata


def search_metadata(
    query_text: str,
    db_dir: Path | str,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Search the metadata database using a text query.
    
    Args:
        query_text: Search query text
        db_dir: Path to vector database directory
        top_k: Number of results to return
        
    Returns:
        DataFrame with top matching metadata entries
    """
    # Load database
    embeddings, metadata = load_vector_db(db_dir)
    
    # Generate query embedding
    query_embedding = embed_texts([query_text])[0]
    
    # Search
    top_indices = similarity_search(query_embedding, embeddings, top_k)
    
    # Return matching metadata
    return metadata.iloc[top_indices]


# ============================================================================
# GROUP SPLITTING
# ============================================================================

KNOWN_GROUP_COLUMNS = ['run_no', 'run', 'chromatography_stage', 'Sample_Code', 'column']


def detect_group_columns(parquet_path: Path) -> List[str]:
    """Read column names from a parquet file and return known group columns present."""
    try:
        df = pd.read_parquet(parquet_path, columns=[])  # schema only — zero rows
        cols = set(df.columns)
    except Exception:
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(parquet_path)
            cols = set(schema.names)
        except Exception:
            return []
    return [c for c in KNOWN_GROUP_COLUMNS if c in cols]


def split_by_groups(
    parquet_path: Path,
    output_dir: Path,
    group_columns: List[str],
) -> Dict[str, Any]:
    """Split a parquet file into sub-files by group columns.

    Writes one parquet per unique combination of *group_columns* to
    ``output_dir/`` and a ``splits_manifest.json`` summarising the splits.
    """
    import json as _json

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)

    # Filter to columns that actually exist
    group_cols = [c for c in group_columns if c in df.columns]
    if not group_cols:
        logger.info(f"No group columns found in {parquet_path.name} — skipping split")
        return {}

    splits: List[Dict[str, Any]] = []
    unique_values: Dict[str, List] = {c: sorted(df[c].dropna().unique().tolist()) for c in group_cols}

    for keys, sub_df in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        # Build filename from group values
        parts = []
        row_meta: Dict[str, Any] = {}
        for col, val in zip(group_cols, keys):
            safe_val = re.sub(r'[^\w-]', '_', str(val))
            parts.append(f"{col}_{safe_val}")
            row_meta[col] = val if not pd.isna(val) else None
        fname = "__".join(parts) + ".parquet"
        out_path = output_dir / fname
        sub_df.to_parquet(out_path, index=False)
        splits.append({
            "file": fname,
            "row_count": len(sub_df),
            **row_meta,
        })

    manifest = {
        "source_file": parquet_path.name,
        "group_columns": group_cols,
        "splits": splits,
        "total_rows": len(df),
        "unique_values": {k: [str(v) for v in vals] for k, vals in unique_values.items()},
    }
    manifest_path = output_dir / "splits_manifest.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info(
        f"Split {parquet_path.name} into {len(splits)} groups by {group_cols} → {output_dir}"
    )
    return manifest


# ============================================================================
# MAIN DATABASE BUILDER
# ============================================================================

def build_database(
    data_dir: Path | str = "Data",
    db_dir: Path | str = "Database"
) -> Dict[str, Any]:
    """
    Main entry point to build the database from data files.
    
    This function:
    1. Discovers all data files in the data directory
    2. Classifies them as metadata or raw data
    3. Saves raw data as Parquet files
    4. Builds a vectorized metadata database
    
    Args:
        data_dir: Path to directory containing input data files
        db_dir: Path to directory where database will be created
        
    Returns:
        Dictionary with summary information about the build process
    """
    data_path = Path(data_dir)
    db_path = Path(db_dir)
    
    logger.info("\n" + "="*70)
    logger.info("DATABASE BUILDER - CHROMATOGRAPHY/MASS SPECTROMETRY DATA")
    logger.info("="*70)
    logger.info(f"Data directory: {data_path.resolve()}")
    logger.info(f"Database directory: {db_path.resolve()}")
    logger.info("="*70 + "\n")
    
    # Step 1: Discover files
    files = find_data_files(data_path)
    
    # Step 2: Classify files
    metadata_files, raw_data_files = split_metadata_and_raw(files)
    
    # Step 3: Process raw data
    logger.info("\n" + "="*70)
    logger.info("PROCESSING RAW DATA FILES")
    logger.info("="*70)
    
    raw_output_dir = db_path / 'raw'
    saved_raw_files = load_and_save_raw_to_parquet(raw_data_files, raw_output_dir)
    
    # Step 4: Build vectorized metadata database
    if metadata_files:
        metadata_output_dir = db_path / 'metadata'
        vector_db_path = build_vectorised_metadata_db(metadata_files, metadata_output_dir)
    else:
        logger.warning("No metadata files found - skipping vector database creation")
        vector_db_path = None
    
    # Step 5: Split raw data by experimental groups
    splits_root = db_path / 'splits'
    splits_created = 0
    for table_name, pq_path in saved_raw_files.items():
        group_cols = detect_group_columns(pq_path)
        if group_cols:
            split_by_groups(pq_path, splits_root / table_name, group_cols)
            splits_created += 1
        else:
            logger.info(f"No group columns detected in {table_name} — skipping split")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("DATABASE BUILD COMPLETE")
    logger.info("="*70)
    logger.info(f"✓ Processed {len(raw_data_files)} raw data files")
    logger.info(f"✓ Saved {len(saved_raw_files)} Parquet files to {raw_output_dir}")
    logger.info(f"✓ Split {splits_created} tables by experimental groups")
    if vector_db_path:
        logger.info(f"✓ Built vector database in {vector_db_path}")
    logger.info("="*70 + "\n")
    
    return {
        'data_dir': str(data_path),
        'db_dir': str(db_path),
        'raw_files_processed': len(raw_data_files),
        'metadata_files_processed': len(metadata_files),
        'raw_output_dir': str(raw_output_dir),
        'vector_db_dir': str(vector_db_path) if vector_db_path else None,
        'saved_tables': list(saved_raw_files.keys()),
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Build the database with default paths
    result = build_database(
        data_dir="Data",
        db_dir="Database"
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BUILD SUMMARY")
    print("="*70)
    for key, value in result.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")
    print("="*70)
