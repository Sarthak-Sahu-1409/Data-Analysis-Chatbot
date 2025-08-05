import logging
import os
from typing import List, Dict, Any, Tuple

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config

logger = logging.getLogger(__name__)


class FileIngestor:
    """
    Handles loading data from various file types, cleaning it, and splitting 
    it into manageable, context-rich chunks for embedding.
    """

    def __init__(
        self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """
        Initializes the FileIngestor.

        Args:
            chunk_size (int): The target size for each text chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        # This text splitter is kept for potential future use or for non-DataFrame sources.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs comprehensive data cleaning and typing on the dataframe.
        - Trims whitespace and normalizes string columns
        - Infers and fixes data types
        - Handles missing values intelligently
        - Removes duplicates
        - Standardizes date formats
        - Handles outliers in numeric columns
        """
        logger.info("Starting enhanced data cleaning...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        for col in df.columns:
            # Get sample of non-null values
            sample = df[col].dropna().head(100)
            
            # Try to infer better data type
            if df[col].dtype == 'object':
                # Check if it's a date
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted {col} to datetime")
                    continue
                except (ValueError, TypeError):
                    pass
                
                # Check if it's numeric but stored as string
                if sample.str.match(r'^-?\d*\.?\d+$').all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                else:
                    # String cleaning
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].str.normalize('NFKC')  # Normalize Unicode characters
                    
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # Calculate statistics for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Fill missing values with median for numeric columns
                df[col].fillna(df[col].median(), inplace=True)
            
            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Fill missing dates with the median date
                df[col].fillna(df[col].median(), inplace=True)
            
            # Handle categorical/string columns
            else:
                # For string columns, fill missing with 'Unknown'
                # but only if more than 5% of the column is populated
                if df[col].count() / len(df) > 0.05:
                    df[col].fillna('Unknown', inplace=True)
                else:
                    # If column is mostly empty, fill with 'Not Available'
                    df[col].fillna('Not Available', inplace=True)
        
        # Remove exact duplicates
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        logger.info("Enhanced data cleaning complete")
        return df

    def ingest_file(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Loads a file, performs enhanced cleaning and preprocessing, and splits into context-rich chunks.
        
        Features:
        - Intelligent data type inference and cleaning
        - Rich context with column types and statistics
        - Enhanced chunk formatting with headers and data types
        - Comprehensive metadata generation
        
        Args:
            file_path (str): The path to the file to ingest.
            
        Returns:
            A tuple containing:
            - A list of text chunks (with summary first)
            - A dictionary of metadata about the file
        """
        logger.info(f"Starting enhanced ingestion for file: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Load and clean data
            if ext == ".csv":
                df = self._load_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                logger.info(f"Successfully loaded Excel file: {file_path}")
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Apply enhanced cleaning and preprocessing
            df = self._clean_data(df)
            
            # Generate comprehensive metadata with statistics
            metadata = {
                "file_name": os.path.basename(file_path),
                "file_type": ext,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude=['int64', 'float64', 'datetime64']).columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            }
            
            # Create an enhanced summary with statistical insights
            df_stats = df.describe(include='all')
            df_stats.fillna('N/A', inplace=True)
            summary_parts = [
                f"Dataset Summary for '{metadata['file_name']}':",
                f"- Total Rows: {metadata['total_rows']:,}",
                f"- Total Columns: {metadata['total_columns']}",
                f"- Column Types:",
                f"  * Numeric: {', '.join(metadata['numeric_columns']) if metadata['numeric_columns'] else 'None'}",
                f"  * DateTime: {', '.join(metadata['datetime_columns']) if metadata['datetime_columns'] else 'None'}",
                f"  * Categorical: {', '.join(metadata['categorical_columns']) if metadata['categorical_columns'] else 'None'}"
            ]
            
            # Add detailed column statistics
            summary_parts.append("\nDetailed Column Analysis:")
            for col in df.columns:
                col_stats = [f"\n{col} ({metadata['data_types'][col]}):"]
                
                # Add type-specific statistics
                if col in metadata['numeric_columns']:
                    stats = df[col].describe()
                    col_stats.extend([
                        f"  * Range: {stats['min']:,.2f} to {stats['max']:,.2f}",
                        f"  * Mean: {stats['mean']:,.2f}",
                        f"  * Median: {df[col].median():,.2f}",
                        f"  * Standard Deviation: {stats['std']:,.2f}"
                    ])
                elif col in metadata['datetime_columns']:
                    col_stats.extend([
                        f"  * Date Range: {df[col].min()} to {df[col].max()}",
                        f"  * Most Common: {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}"
                    ])
                else:
                    value_counts = df[col].value_counts().head(3)
                    col_stats.append(f"  * Top Values: {', '.join([f'{v} ({c:,} times)' for v, c in value_counts.items()])}")
                
                col_stats.extend([
                    f"  * Unique Values: {metadata['unique_counts'][col]:,}",
                    f"  * Missing Values: {metadata['null_counts'][col]:,} ({(metadata['null_counts'][col]/len(df))*100:.1f}%)"
                ])
                summary_parts.extend(col_stats)
            
            summary = "\n".join(summary_parts)

            # Convert DataFrame rows to context-rich strings with type and category information
            rows_as_strings = []
            for _, row in df.iterrows():
                row_parts = []
                for k, v in row.to_dict().items():
                    # Add type information and formatting based on data type
                    if pd.isna(v):
                        formatted_value = "Not Available"
                    elif pd.api.types.is_numeric_dtype(type(v)):
                        formatted_value = f"{v:,.2f}" if isinstance(v, float) else str(v)
                        row_parts.append(f"{k} (numeric): {formatted_value}")
                    elif pd.api.types.is_datetime64_any_dtype(type(v)):
                        formatted_value = v.strftime("%Y-%m-%d %H:%M:%S")
                        row_parts.append(f"{k} (date): {formatted_value}")
                    else:
                        formatted_value = f"'{v}'" if isinstance(v, str) else str(v)
                        row_parts.append(f"{k} (category): {formatted_value}")
                rows_as_strings.append(", ".join(row_parts))
            
            # Prepare enhanced headers with column types
            header_parts = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    header_parts.append(f"{col} (numeric)")
                elif pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                    header_parts.append(f"{col} (date)")
                else:
                    header_parts.append(f"{col} (category)")
            headers = ", ".join(header_parts)
            
            # Split rows into chunks with enhanced context
            chunks = []
            current_chunk_rows = []
            
            for row_str in rows_as_strings:
                # Check if adding the new row would exceed the chunk size
                potential_chunk = "\n".join(current_chunk_rows + [row_str])
                if len(headers) + 1 + len(potential_chunk) > config.CHUNK_SIZE and current_chunk_rows:
                    # Finalize current chunk with enhanced context
                    context_header = (
                        f"Column Details:\n{headers}\n\n"
                        f"Data Preview:\n"
                    )
                    final_chunk_content = "\n".join(current_chunk_rows)
                    chunks.append(f"{context_header}{final_chunk_content}")
                    current_chunk_rows = [row_str]
                else:
                    current_chunk_rows.append(row_str)
            
            # Add the last chunk if any rows remain
            if current_chunk_rows:
                context_header = (
                    f"Column Details:\n{headers}\n\n"
                    f"Data Preview:\n"
                )
                final_chunk_content = "\n".join(current_chunk_rows)
                chunks.append(f"{context_header}{final_chunk_content}")
            
            # Add the detailed summary as the first chunk
            chunks.insert(0, summary)
            
            logger.info(f"Split file into {len(chunks)} chunks (including summary chunk)")
            metadata["num_chunks"] = len(chunks)
            logger.info(f"Enhanced ingestion complete for: {file_path}")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Failed to ingest file '{file_path}': {e}")
            raise

            # Prepare headers string to be prepended to each chunk
            headers = ", ".join(df.columns)

                # Convert DataFrame rows to context-rich strings with type and category information
            rows_as_strings = []
            for _, row in df.iterrows():
                row_parts = []
                for k, v in row.to_dict().items():
                    # Add type information and formatting based on data type
                    if pd.isna(v):
                        formatted_value = "Not Available"
                    elif pd.api.types.is_numeric_dtype(type(v)):
                        formatted_value = f"{v:,.2f}" if isinstance(v, float) else str(v)
                        row_parts.append(f"{k} (numeric): {formatted_value}")
                    elif pd.api.types.is_datetime64_any_dtype(type(v)):
                        formatted_value = v.strftime("%Y-%m-%d %H:%M:%S")
                        row_parts.append(f"{k} (date): {formatted_value}")
                    else:
                        formatted_value = f"'{v}'" if isinstance(v, str) else str(v)
                        row_parts.append(f"{k} (category): {formatted_value}")
                rows_as_strings.append(", ".join(row_parts))
            
            # Split rows into chunks with enhanced context
            chunks = []
            current_chunk_rows = []
            
            # Process rows into chunks with column type information
            for row_str in rows_as_strings:
                # Check if adding the new row would exceed the chunk size
                potential_chunk = "\n".join(current_chunk_rows + [row_str])
                if len(headers) + 1 + len(potential_chunk) > config.CHUNK_SIZE and current_chunk_rows:
                    # Finalize current chunk with enhanced context
                    context_header = (
                        f"Column Details:\n{headers}\n\n"
                        f"Data Preview:\n"
                    )
                    final_chunk_content = "\n".join(current_chunk_rows)
                    chunks.append(f"{context_header}{final_chunk_content}")
                    current_chunk_rows = [row_str]
                else:
                    current_chunk_rows.append(row_str)
            
            # Add the last chunk if any rows remain
            if current_chunk_rows:
                context_header = (
                    f"Column Details:\n{headers}\n\n"
                    f"Data Preview:\n"
                )
                final_chunk_content = "\n".join(current_chunk_rows)
                chunks.append(f"{context_header}{final_chunk_content}")
            
            # Add the detailed summary as the first chunk
            chunks.insert(0, summary)
            
            logger.info(f"Split file into {len(chunks)} chunks (including summary chunk)")
            metadata["num_chunks"] = len(chunks)
            logger.info(f"Enhanced ingestion complete for: {file_path}")
            return chunks, metadata

        except Exception as e:
            logger.error(f"Failed to ingest file '{file_path}': {e}")
            raise

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV file, attempting multiple common encodings.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded dataframe.
            
        Raises:
            UnicodeDecodeError: If none of the encodings work.
        """
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                if encoding == encodings[-1]:
                    logger.error(f"Failed to load CSV with any encoding: {encodings}")
                    raise
                logger.warning(f"{encoding} encoding failed, trying next encoding")
                continue
