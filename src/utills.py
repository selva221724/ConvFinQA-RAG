import re
from typing import List, Dict, Any, Optional
from math import *

# LangChain imports
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.retrievers import BaseRetriever

# Optional imports for advanced features
TORCH_AVAILABLE = True
SENTENCE_TRANSFORMERS_AVAILABLE = True

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not installed. Advanced reranking features will be disabled.")
    print("To enable, run: pip install torch")

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Advanced reranking will be disabled.")
    print("To enable, run: pip install sentence-transformers")
    CrossEncoder = None

class AdvancedTableParser:
    """Enhanced table parser with improved numerical context extraction"""
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with enhanced table representation and numerical context"""
        parsed_documents = []
        
        for doc in documents:
            content = doc.page_content
            
            # Check if document contains a table
            if "Table:" in content or doc.metadata.get("has_table", False):
                # Extract table from markdown format
                table_pattern = r"Table:\n(\|.+\|\n)+"
                table_matches = re.findall(table_pattern, content, re.MULTILINE)
                
                if table_matches:
                    # Process each table found in the document
                    for table_match in table_matches:
                        table_text = "Table:\n" + table_match
                        
                        # Convert to structured format
                        structured_table = self._parse_markdown_table(table_text)
                        
                        # Enhance content with explicit column descriptions
                        if structured_table:
                            enhanced_description = self._generate_enhanced_table_description(structured_table)
                            content = content.replace(table_text, enhanced_description)
            
            # Extract and add numerical entities from text
            numerical_entities = self._extract_numerical_entities(content)
            if numerical_entities and len(numerical_entities) > 0:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["numerical_entities"] = numerical_entities
            
            doc.page_content = content
            parsed_documents.append(doc)
        
        return parsed_documents
    
    def _parse_markdown_table(self, table_text: str) -> List[Dict[str, Any]]:
        """Parse markdown table into structured format"""
        lines = table_text.strip().split('\n')
        
        # Remove "Table:" header if present
        if lines[0].startswith("Table:"):
            lines = lines[1:]
        
        if not lines:
            return []
        
        # Parse header
        header = lines[0].strip('|').split('|')
        header = [h.strip() for h in header]
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            if '|' not in line:  # Skip separator lines
                continue
            
            row_values = line.strip('|').split('|')
            row_values = [v.strip() for v in row_values]
            
            # Create row dict
            row_dict = {}
            for i, col in enumerate(header):
                if i < len(row_values):
                    row_dict[col] = row_values[i]
            
            rows.append(row_dict)
        
        return rows
    
    def _generate_enhanced_table_description(self, structured_table: List[Dict[str, Any]]) -> str:
        """Generate a detailed description of the table"""
        if not structured_table:
            return ""
        
        columns = list(structured_table[0].keys())
        description = f"Table with {len(structured_table)} rows and {len(columns)} columns: {', '.join(columns)}\n"
        
        # Add sample of numerical data
        numerical_cols = []
        for col in columns:
            try:
                val = structured_table[0][col].replace(',', '')
                float(val)
                numerical_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        if numerical_cols:
            sample_row = structured_table[0]
            num_samples = ", ".join([f"{col}: {sample_row[col]}" for col in numerical_cols[:3]])
            description += f"Numerical data sample: {num_samples}\n"
        
        return description
    
    def _extract_numerical_entities(self, text: str) -> List[str]:
        """Extract numerical entities from text"""
        entities = []
        number_pattern = r'(\$?)([\d,]+\.?\d*)(\s*%?)'
        matches = re.finditer(number_pattern, text)
        
        for match in matches:
            currency_symbol = match.group(1)
            number_str = match.group(2)
            percentage = match.group(3).strip()
            
            clean_number = number_str.replace(',', '')
            if not clean_number:
                continue
                
            try:
                float_value = float(clean_number)
                entity_type = "number"
                if currency_symbol:
                    entity_type = "currency"
                elif percentage:
                    entity_type = "percentage"
                
                entity_str = f"{float_value} ({entity_type})"
                entities.append(entity_str)
            except ValueError:
                continue
        
        return entities

class QueryExpander:
    """Expands queries to improve retrieval performance"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name)
    
    def expand_query(self, query: str) -> str:
        """Expand a query to improve retrieval performance"""
        expansion_prompt = f"""
        You are a financial expert. Expand the following financial query to make it more comprehensive and precise.
        
        Original Query: {query}
        
        Provide an expanded version that:
        1. Clarifies any ambiguous financial terms
        2. Adds relevant financial context
        3. Includes alternative phrasings for key concepts
        4. Specifies any numerical relationships that might be relevant
        """
        
        try:
            expanded_query = self.llm.invoke(expansion_prompt).content
            return f"{query}\n\nExpanded Query: {expanded_query}"
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query

class CrossEncoderReranker:
    """Reranks documents using a cross-encoder model"""
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code: bool = False):
        self.model = None
        
        if not TORCH_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: CrossEncoderReranker requires torch and sentence-transformers.")
            return
        
        try:
            self.model = CrossEncoder(model_name, max_length=512, trust_remote_code=trust_remote_code)
        except Exception as e:
            print(f"Error loading cross-encoder model: {e}")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on relevance to query"""
        if self.model is None or not documents:
            print("Using basic retrieval (reranker not available)")
            return documents[:top_k] if documents else []
        
        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)
            doc_score_pairs = list(zip(documents, scores))
            ranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)]
            return ranked_docs[:top_k]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents[:top_k]

class CustomRetriever:
    """Custom retriever with multi-stage retrieval and reranking"""
    
    def __init__(self, base_retriever, embeddings, reranker=None, top_k: int = 5):
        self.base_retriever = base_retriever
        self.embeddings = embeddings
        self.reranker = reranker
        self.top_k = top_k
        
        try:
            embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever
            )
            self.use_compression = True
        except Exception as e:
            print(f"Warning: Using basic retrieval due to: {e}")
            self.use_compression = False
    
    def invoke(self, query: str) -> List[Document]:
        """Get relevant documents using multi-stage retrieval"""
        try:
            # First stage: Get documents
            documents = (self.compression_retriever.invoke(query) if self.use_compression 
                       else self.base_retriever.invoke(query))
            
            # Second stage: Rerank if available
            if self.reranker is not None:
                try:
                    documents = self.reranker.rerank(query, documents, self.top_k)
                except Exception as e:
                    print(f"Reranking failed: {e}")
                    documents = documents[:self.top_k]
            else:
                documents = documents[:self.top_k]
            
            return documents
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return self.base_retriever.invoke(query)[:self.top_k]
