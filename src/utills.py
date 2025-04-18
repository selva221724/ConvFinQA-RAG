
import re
from typing import List, Dict, Any, Optional, Tuple
from math import *

# LangChain imports
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from typing import Optional

# Optional imports for advanced features
TORCH_AVAILABLE = True
SENTENCE_TRANSFORMERS_AVAILABLE = True

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not installed. Advanced reranking features will be disabled.")
    print("To enable, run: pip install torch")

# Import for cross-encoder reranking
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Advanced reranking will be disabled.")
    print("To enable, run: pip install sentence-transformers")
    CrossEncoder = None

class TableParser:
    """Extracts and formats tables for better numerical reasoning"""
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents to enhance table representation"""
        parsed_documents = []
        
        for doc in documents:
            content = doc.page_content
            
            # Check if document contains a table
            if doc.metadata.get("has_table", False):
                # Extract table from markdown format
                table_pattern = r"Table:\n(\|.+\|\n)+"
                table_matches = re.search(table_pattern, content)
                
                if table_matches:
                    table_text = table_matches.group(0)
                    
                    # Convert to structured format
                    structured_table = self._parse_markdown_table(table_text)
                    
                    # Enhance content with explicit column descriptions
                    if structured_table:
                        table_description = self._generate_table_description(structured_table)
                        content = content.replace(table_text, table_description)
                        doc.page_content = content
                        
                        # Instead of storing the complex structured_table object,
                        # just store some simple metadata about the table
                        if not doc.metadata:
                            doc.metadata = {}
                        
                        # Store simple metadata about the table (only primitive types)
                        if structured_table and len(structured_table) > 0:
                            doc.metadata["table_rows"] = len(structured_table)
                            doc.metadata["table_columns"] = len(structured_table[0].keys()) if structured_table[0] else 0
                            doc.metadata["table_columns_names"] = ", ".join(structured_table[0].keys()) if structured_table[0] else ""
            
            parsed_documents.append(doc)
        
        return parsed_documents
    
    def _parse_markdown_table(self, table_text: str) -> List[Dict[str, Any]]:
        """Parse markdown table into structured format"""
        lines = table_text.strip().split('\n')
        
        # Remove "Table:" header if present
        if lines[0].startswith("Table:"):
            lines = lines[1:]
        
        # Extract header and rows
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
    
    def _generate_table_description(self, structured_table: List[Dict[str, Any]]) -> str:
        """Generate a concise textual description of the table for better LLM understanding"""
        if not structured_table:
            return ""
        
        # Get columns
        columns = list(structured_table[0].keys())
        
        # Create a more compact description
        description = f"Table: {len(structured_table)} rows with columns: {', '.join(columns)}\n"
        
        # Add sample of numerical data (only first row)
        numerical_cols = []
        for col in columns:
            # Check if column contains numbers
            try:
                val = structured_table[0][col].replace(',', '')
                float(val)
                numerical_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        if numerical_cols:
            # Only include a sample from the first row to save space
            sample_row = structured_table[0]
            num_samples = ", ".join([f"{col}: {sample_row[col]}" for col in numerical_cols[:3]])
            description += f"Numerical data sample: {num_samples}\n"
        
        return description


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
                        
                        # Enhance content with explicit column descriptions and numerical analysis
                        if structured_table:
                            enhanced_description = self._generate_enhanced_table_description(structured_table)
                            content = content.replace(table_text, enhanced_description)
                            
                            # Extract numerical relationships from the table
                            numerical_insights = self._extract_numerical_insights(structured_table)
                            
                            # Add numerical insights to metadata
                            if not doc.metadata:
                                doc.metadata = {}
                            doc.metadata.update(numerical_insights)
            
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
        """Parse markdown table into structured format with enhanced numerical detection"""
        lines = table_text.strip().split('\n')
        
        # Remove "Table:" header if present
        if lines[0].startswith("Table:"):
            lines = lines[1:]
        
        # Extract header and rows
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
                    # Try to convert numerical values
                    value = row_values[i]
                    try:
                        # Remove commas from numbers
                        clean_value = value.replace(',', '')
                        if '%' in clean_value:
                            # Handle percentage values
                            clean_value = clean_value.replace('%', '')
                            row_dict[col] = {"value": float(clean_value), "type": "percentage"}
                        else:
                            # Try to convert to float
                            row_dict[col] = {"value": float(clean_value), "type": "number"}
                    except (ValueError, TypeError):
                        # Keep as string if not a number
                        row_dict[col] = {"value": value, "type": "text"}
            
            rows.append(row_dict)
        
        return rows
    
    def _generate_enhanced_table_description(self, structured_table: List[Dict[str, Any]]) -> str:
        """Generate a detailed description of the table with numerical analysis"""
        if not structured_table:
            return ""
        
        # Get columns
        columns = list(structured_table[0].keys())
        
        # Create a more detailed description
        description = f"Table with {len(structured_table)} rows and {len(columns)} columns: {', '.join(columns)}\n"
        
        # Identify numerical columns
        numerical_cols = []
        for col in columns:
            # Check if column contains numbers
            if structured_table[0][col]["type"] in ["number", "percentage"]:
                numerical_cols.append(col)
        
        # Add detailed numerical information
        if numerical_cols:
            description += "Numerical columns:\n"
            for col in numerical_cols:
                # Calculate min, max, average for numerical columns
                values = [row[col]["value"] for row in structured_table if row[col]["type"] in ["number", "percentage"]]
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) / len(values)
                    
                    # Format based on type
                    if structured_table[0][col]["type"] == "percentage":
                        description += f"  - {col}: Range {min_val:.2f}% to {max_val:.2f}%, Average: {avg_val:.2f}%\n"
                    else:
                        description += f"  - {col}: Range {min_val:.2f} to {max_val:.2f}, Average: {avg_val:.2f}\n"
        
        return description
    
    def _extract_numerical_insights(self, structured_table: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract numerical insights from the table for metadata"""
        if not structured_table:
            return {}
        
        columns = list(structured_table[0].keys())
        numerical_insights = {
            "numerical_columns": [],
            "column_statistics": {}
        }
        
        for col in columns:
            # Check if column contains numbers
            if structured_table[0][col]["type"] in ["number", "percentage"]:
                numerical_insights["numerical_columns"].append(col)
                
                # Calculate statistics
                values = [row[col]["value"] for row in structured_table if row[col]["type"] in ["number", "percentage"]]
                if values:
                    numerical_insights["column_statistics"][col] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "type": structured_table[0][col]["type"]
                    }
        
        return numerical_insights
    
    def _extract_numerical_entities(self, text: str) -> List[str]:
        """Extract numerical entities from text with simplified format for Pinecone"""
        entities = []
        
        # Pattern for numbers with optional commas and decimal points
        number_pattern = r'(\$?)([\d,]+\.?\d*)(\s*%?)'
        
        # Find all matches
        matches = re.finditer(number_pattern, text)
        
        for match in matches:
            currency_symbol = match.group(1)
            number_str = match.group(2)
            percentage = match.group(3).strip()
            
            # Clean the number string
            clean_number = number_str.replace(',', '')
            
            # Skip empty strings
            if not clean_number:
                continue
                
            try:
                # Convert to float
                float_value = float(clean_number)
                
                # Determine the type
                entity_type = "number"
                if currency_symbol:
                    entity_type = "currency"
                elif percentage:
                    entity_type = "percentage"
                
                # Create a simplified string representation
                entity_str = f"{float_value} ({entity_type})"
                entities.append(entity_str)
            except ValueError:
                # Skip if conversion to float fails
                continue
        
        return entities


class QueryExpander:
    """Expands queries to improve retrieval performance"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize with specified model"""
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
        
        Your expansion should help in retrieving more relevant financial documents.
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
        """Initialize with specified model"""
        self.model = None
        
        # Check if required dependencies are available
        if not TORCH_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: CrossEncoderReranker requires torch and sentence-transformers.")
            print("Advanced reranking will be disabled.")
            print("To enable, run: pip install torch sentence-transformers")
            return
        
        try:
            self.model = CrossEncoder(model_name, max_length=512, trust_remote_code=trust_remote_code)
            print(f"Successfully loaded cross-encoder model: {model_name}")
        except Exception as e:
            print(f"Error loading cross-encoder model: {e}")
            if not trust_remote_code and "trust_remote_code=True" in str(e):
                print("Trying with trust_remote_code=True...")
                try:
                    self.model = CrossEncoder(model_name, max_length=512, trust_remote_code=True)
                    print(f"Successfully loaded cross-encoder model with trust_remote_code=True: {model_name}")
                except Exception as e2:
                    print(f"Error loading cross-encoder model with trust_remote_code=True: {e2}")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on relevance to query"""
        if self.model is None or not documents:
            # If model is not available, just return top_k documents without reranking
            print("Using basic retrieval (reranker not available)")
            return documents[:top_k] if documents else []
        
        # Prepare document-query pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        try:
            # Get scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Create document-score pairs
            doc_score_pairs = list(zip(documents, scores))
            
            # Sort by score in descending order
            ranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)]
            
            # Return top_k documents
            return ranked_docs[:top_k]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents[:top_k]


class CustomRetriever:
    """A custom retriever that implements multi-stage retrieval with reranking"""
    
    def __init__(
        self, 
        base_retriever, 
        embeddings, 
        reranker=None, 
        top_k: int = 5
    ):
        """Initialize with base retriever and optional reranker"""
        self.base_retriever = base_retriever
        self.embeddings = embeddings
        self.reranker = reranker
        self.top_k = top_k
        self.compression_retriever = None
        self.use_compression = False
        
        # Check if reranker is available
        if self.reranker is None and not TORCH_AVAILABLE:
            print("Note: Advanced reranking is disabled. Using basic retrieval only.")
            print("To enable advanced reranking, install torch and sentence-transformers.")
        
        # Create embedding filter
        try:
            embeddings_filter = EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=0.7
            )
            
            # Create compression retriever
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever
            )
            self.use_compression = True
        except Exception as e:
            print(f"Warning: Could not initialize compression retriever: {e}")
            print("Falling back to basic retrieval")
            self.use_compression = False
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using multi-stage retrieval (deprecated method)"""
        # Call the new invoke method for backward compatibility
        return self.invoke(query)
    
    def invoke(self, query: str) -> List[Document]:
        """Get relevant documents using multi-stage retrieval"""
        try:
            # First stage: Get documents from retriever
            if self.use_compression:
                # Use compression retriever if available
                documents = self.compression_retriever.invoke(query)
            else:
                # Fall back to base retriever
                documents = self.base_retriever.invoke(query)
            
            # Second stage: Apply cross-encoder reranking if available
            if self.reranker is not None:
                try:
                    documents = self.reranker.rerank(query, documents, self.top_k)
                except Exception as e:
                    print(f"Reranking failed: {e}")
                    print("Falling back to basic retrieval")
                    documents = documents[:self.top_k]
            else:
                documents = documents[:self.top_k]
            
            return documents
        except Exception as e:
            print(f"Error in multi-stage retrieval: {e}")
            print("Falling back to basic retrieval")
            # Last resort fallback
            try:
                return self.base_retriever.invoke(query)[:self.top_k]
            except:
                # If invoke fails, try the deprecated method as a last resort
                return self.base_retriever.get_relevant_documents(query)[:self.top_k]


class AdvancedCalculatorTool:
    """Advanced calculator with enhanced numerical reasoning capabilities"""
    
    def calculate(self, query: str, documents: List[Document], calculation_request: Optional[str] = None) -> Dict[str, Any]:
        """Process calculation requests with enhanced numerical reasoning"""
        
        if not calculation_request:
            # Try to automatically extract calculation request from query
            extracted_request = self._extract_calculation_request(query)
            if extracted_request:
                calculation_request = extracted_request
            else:
                return {
                    "calculation_result": None,
                    "calculation_performed": False
                }
        
        try:
            # Clean the calculation request
            clean_request = calculation_request.strip()
            
            # Extract numerical entities from documents
            numerical_entities = self._extract_numerical_entities_from_docs(documents)
            
            # Handle different calculation types
            if "%" in clean_request or "percent" in clean_request.lower():
                result = self._handle_percentage_calculation(clean_request, numerical_entities)
            elif any(term in clean_request.lower() for term in ["growth", "increase", "decrease", "change"]):
                result = self._handle_change_calculation(clean_request, numerical_entities)
            elif any(term in clean_request.lower() for term in ["average", "mean", "median"]):
                result = self._handle_statistical_calculation(clean_request, numerical_entities)
            else:
                result = self._handle_arithmetic_calculation(clean_request)
            
            return {
                "calculation_result": result,
                "calculation_performed": True,
                "calculation_request": clean_request,
                "numerical_entities_used": numerical_entities[:5] if numerical_entities else []
            }
            
        except Exception as e:
            return {
                "calculation_result": f"Error: {str(e)}",
                "calculation_performed": False,
                "calculation_request": calculation_request
            }
    
    def _extract_calculation_request(self, query: str) -> Optional[str]:
        """Automatically extract calculation request from query"""
        # Patterns that indicate calculations
        calculation_patterns = [
            r"calculate\s+(.+)",
            r"what is\s+(.+)",
            r"find\s+(.+)",
            r"compute\s+(.+)",
            r"determine\s+(.+)"
        ]
        
        for pattern in calculation_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_numerical_entities_from_docs(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract numerical entities from documents"""
        entities = []
        
        for doc in documents:
            # Check if document has pre-extracted numerical entities
            if doc.metadata and "numerical_entities" in doc.metadata:
                entities.extend(doc.metadata["numerical_entities"])
                continue
            
            # Otherwise extract from content
            content = doc.page_content
            
            # Pattern for numbers with optional commas and decimal points
            number_pattern = r'(\$?)([\d,]+\.?\d*)(\s*%?)'
            
            # Find all matches
            matches = re.finditer(number_pattern, content)
            
            for match in matches:
                currency_symbol = match.group(1)
                number_str = match.group(2)
                percentage = match.group(3).strip()
                
                # Clean the number string
                clean_number = number_str.replace(',', '')
                
                # Skip empty strings
                if not clean_number:
                    continue
                
                try:
                    # Convert to float
                    float_value = float(clean_number)
                    
                    # Determine the type
                    entity_type = "number"
                    if currency_symbol:
                        entity_type = "currency"
                    elif percentage:
                        entity_type = "percentage"
                    
                    # Get surrounding context (up to 30 chars before and after)
                    start_pos = max(0, match.start() - 30)
                    end_pos = min(len(content), match.end() + 30)
                    context = content[start_pos:end_pos]
                    
                    # Add to entities
                    entities.append({
                        "value": float_value,
                        "original": match.group(0),
                        "type": entity_type,
                        "context": context
                    })
                except ValueError:
                    # Skip if conversion to float fails
                    continue
        
        return entities
    
    def _handle_percentage_calculation(self, request: str, numerical_entities: List[Dict[str, Any]]) -> str:
        """Handle percentage calculations with enhanced context awareness"""
        # Common percentage calculation patterns
        increase_pattern = r"(?:increase|growth|change).*?from\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s+to\s+(\d+(?:,\d{3})*(?:\.\d+)?)"
        percentage_of_pattern = r"(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:,\d{3})*(?:\.\d+)?)"
        
        # Check for increase/decrease pattern
        increase_match = re.search(increase_pattern, request)
        if increase_match:
            start_val = float(increase_match.group(1).replace(',', ''))
            end_val = float(increase_match.group(2).replace(',', ''))
            
            if start_val == 0:
                return "Cannot calculate percentage change from zero"
            
            percent_change = ((end_val - start_val) / start_val) * 100
            return f"{percent_change:.2f}%"
        
        # Check for percentage of pattern
        percent_of_match = re.search(percentage_of_pattern, request)
        if percent_of_match:
            percent = float(percent_of_match.group(1))
            base = float(percent_of_match.group(2).replace(',', ''))
            
            result = (percent / 100) * base
            return f"{result:.2f}"
        
        # If no pattern matched, try to use numerical entities
        if numerical_entities:
            percentage_entities = [e for e in numerical_entities if e["type"] == "percentage"]
            if percentage_entities:
                # Use the first percentage entity
                return f"Using extracted percentage: {percentage_entities[0]['original']} from context: '{percentage_entities[0]['context']}'"
        
        # If no pattern matched, try to evaluate as expression
        return self._handle_arithmetic_calculation(request)
    
    def _handle_change_calculation(self, request: str, numerical_entities: List[Dict[str, Any]]) -> str:
        """Handle calculations involving change over time or between values"""
        # Try to find two numerical values in the request
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', request)
        
        if len(numbers) >= 2:
            # Use the first two numbers found
            start_val = float(numbers[0].replace(',', ''))
            end_val = float(numbers[1].replace(',', ''))
            
            if "percent" in request.lower() or "%" in request:
                # Calculate percentage change
                if start_val == 0:
                    return "Cannot calculate percentage change from zero"
                
                percent_change = ((end_val - start_val) / start_val) * 100
                return f"{percent_change:.2f}%"
            else:
                # Calculate absolute change
                absolute_change = end_val - start_val
                return f"{absolute_change:.2f}"
        
        # If no numbers found in request, try to use numerical entities
        if numerical_entities and len(numerical_entities) >= 2:
            # Use the first two numerical entities
            start_val = numerical_entities[0]["value"]
            end_val = numerical_entities[1]["value"]
            
            if "percent" in request.lower() or "%" in request:
                # Calculate percentage change
                if start_val == 0:
                    return "Cannot calculate percentage change from zero"
                
                percent_change = ((end_val - start_val) / start_val) * 100
                return f"{percent_change:.2f}%"
            else:
                # Calculate absolute change
                absolute_change = end_val - start_val
                return f"{absolute_change:.2f}"
        
        return "Could not identify values for change calculation"
    
    def _handle_statistical_calculation(self, request: str, numerical_entities: List[Dict[str, Any]]) -> str:
        """Handle statistical calculations like average, mean, median"""
        if not numerical_entities:
            return "No numerical entities found for statistical calculation"
        
        # Extract values based on context
        values = [entity["value"] for entity in numerical_entities]
        
        if "average" in request.lower() or "mean" in request.lower():
            # Calculate average
            avg_val = sum(values) / len(values)
            return f"{avg_val:.2f}"
        
        elif "median" in request.lower():
            # Calculate median
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                # Even number of values
                median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            else:
                # Odd number of values
                median = sorted_values[n//2]
            return f"{median:.2f}"
        
        return "Unrecognized statistical calculation"
    
    def _handle_arithmetic_calculation(self, request: str) -> str:
        """Handle basic arithmetic calculations with enhanced parsing"""
        # Replace textual operators with symbols
        request = request.lower()
        request = request.replace("divided by", "/")
        request = request.replace("multiplied by", "*")
        request = request.replace("times", "*")
        request = request.replace("plus", "+")
        request = request.replace("minus", "-")
        
        # Extract the arithmetic expression
        expression_pattern = r"([\d\s\+\-\*\/\(\)\.\,]+)"
        match = re.search(expression_pattern, request)
        
        if not match:
            return "No arithmetic expression found"
        
        expression = match.group(1).strip()
        expression = expression.replace(',', '')  # Remove commas from numbers
        
        # Safely evaluate the expression
        try:
            # Create a safe namespace with only math functions
            safe_namespace = {
                k: v for k, v in globals().items() 
                if k in dir(math) or k in ['__builtins__']
            }
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            
            # Format based on result type
            if isinstance(result, int):
                return str(result)
            else:
                return f"{result:.2f}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


class CalculatorTool:
    """Provides explicit calculation capabilities for numerical reasoning"""
    
    def calculate(self, query: str, documents: List[Document], calculation_request: Optional[str] = None) -> Dict[str, Any]:
        """Process calculation requests"""
        
        if not calculation_request:
            # No explicit calculation requested
            return {
                "calculation_result": None,
                "calculation_performed": False
            }
        
        try:
            # Extract numbers directly from document content instead of metadata
            # This avoids using complex objects in metadata that Pinecone doesn't support
            
            # Clean the calculation request
            clean_request = calculation_request.strip()
            
            # Handle percentage calculations
            if "%" in clean_request or "percent" in clean_request.lower():
                result = self._handle_percentage_calculation(clean_request)
            # Handle basic arithmetic
            else:
                result = self._handle_arithmetic_calculation(clean_request)
            
            return {
                "calculation_result": result,
                "calculation_performed": True,
                "calculation_request": clean_request
            }
            
        except Exception as e:
            return {
                "calculation_result": f"Error: {str(e)}",
                "calculation_performed": False,
                "calculation_request": calculation_request
            }
    
    def _handle_percentage_calculation(self, request: str) -> str:
        """Handle percentage calculations"""
        # Common percentage calculation patterns
        increase_pattern = r"(?:increase|growth|change).*?from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)"
        percentage_of_pattern = r"(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)"
        
        # Check for increase/decrease pattern
        increase_match = re.search(increase_pattern, request)
        if increase_match:
            start_val = float(increase_match.group(1))
            end_val = float(increase_match.group(2))
            
            if start_val == 0:
                return "Cannot calculate percentage change from zero"
            
            percent_change = ((end_val - start_val) / start_val) * 100
            return f"{percent_change:.2f}%"
        
        # Check for percentage of pattern
        percent_of_match = re.search(percentage_of_pattern, request)
        if percent_of_match:
            percent = float(percent_of_match.group(1))
            base = float(percent_of_match.group(2))
            
            result = (percent / 100) * base
            return f"{result:.2f}"
        
        # If no pattern matched, try to evaluate as expression
        return self._handle_arithmetic_calculation(request)
    
    def _handle_arithmetic_calculation(self, request: str) -> str:
        """Handle basic arithmetic calculations"""
        # Replace textual operators with symbols
        request = request.lower()
        request = request.replace("divided by", "/")
        request = request.replace("multiplied by", "*")
        request = request.replace("times", "*")
        request = request.replace("plus", "+")
        request = request.replace("minus", "-")
        
        # Extract the arithmetic expression
        expression_pattern = r"([\d\s\+\-\*\/\(\)\.\,]+)"
        match = re.search(expression_pattern, request)
        
        if not match:
            return "No arithmetic expression found"
        
        expression = match.group(1).strip()
        expression = expression.replace(",", "")  # Remove commas from numbers
        
        # Safely evaluate the expression
        # Using eval with math functions from math module
        try:
            # Create a safe namespace with only math functions
            safe_namespace = {
                k: v for k, v in globals().items() 
                if k in dir(math) or k in ['__builtins__']
            }
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            
            # Format based on result type
            if isinstance(result, int):
                return str(result)
            else:
                return f"{result:.2f}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
