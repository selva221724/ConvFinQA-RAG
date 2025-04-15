import os
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from math import *

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Pinecone
import pinecone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class NumericalTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that preserves numerical context"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Override to add numerical metadata to split documents"""
        split_docs = super().split_documents(documents)
        
        # Enhance documents with numerical metadata
        enhanced_docs = []
        for doc in split_docs:
            # Detect tables in content
            contains_table = self._detect_table(doc.page_content)
            
            # Extract numerical entities
            numerical_entities = self._extract_numerical_entities(doc.page_content)
            
            # Add metadata
            if not doc.metadata:
                doc.metadata = {}
            
            doc.metadata["contains_table"] = contains_table
            doc.metadata["numerical_entities"] = numerical_entities
            doc.metadata["contains_numbers"] = len(numerical_entities) > 0
            
            enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def _detect_table(self, text: str) -> bool:
        """Detect if text likely contains a table"""
        # Simple heuristic: look for patterns of multiple numbers and separators
        # Check for markdown or ascii tables
        table_patterns = [
            r"\|\s*[\d.]+\s*\|",  # Markdown tables with numbers
            r"[\+\-]{3,}",        # ASCII table borders
            r"(\d+\s+){3,}\d+"    # Multiple numbers in sequence
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_numerical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical entities from text"""
        # Match numbers with optional units/currencies
        number_pattern = r'(\$|€|£)?(\d+(?:,\d+)*(?:\.\d+)?)\s*(million|billion|percent|%|k|M|B)?'
        
        matches = re.finditer(number_pattern, text)
        entities = []
        
        for match in matches:
            currency, value, unit = match.groups()
            
            # Clean and convert value
            clean_value = value.replace(',', '')
            
            # Determine multiplier based on unit
            multiplier = 1
            if unit in ['million', 'M']:
                multiplier = 1_000_000
            elif unit in ['billion', 'B']:
                multiplier = 1_000_000_000
            elif unit == 'k':
                multiplier = 1_000
            
            # Create entity
            entity = {
                "value": float(clean_value) * multiplier,
                "raw_text": match.group(0),
                "currency": currency if currency else None,
                "unit": unit if unit else None,
                "position": match.span()
            }
            
            entities.append(entity)
            
        return entities


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
                    
                    # Add structured table to metadata
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["structured_table"] = structured_table
                    
                    # Enhance content with explicit column descriptions
                    if structured_table:
                        table_description = self._generate_table_description(structured_table)
                        content = content.replace(table_text, table_description)
                        doc.page_content = content
            
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
        """Generate a textual description of the table for better LLM understanding"""
        if not structured_table:
            return ""
        
        # Get columns
        columns = list(structured_table[0].keys())
        
        description = "Table with columns: " + ", ".join(columns) + "\n"
        description += f"The table contains {len(structured_table)} rows of data.\n"
        
        # Add sample of numerical data
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
            description += "Numerical columns: " + ", ".join(numerical_cols) + "\n"
            description += "Sample values:\n"
            
            for col in numerical_cols:
                sample_values = [row[col] for row in structured_table[:3]]
                description += f"- {col}: {', '.join(sample_values)}\n"
        
        return description


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
            # Extract numbers from documents for reference
            all_numbers = []
            for doc in documents:
                if "numerical_entities" in doc.metadata:
                    all_numbers.extend(doc.metadata["numerical_entities"])
            
            # Clean the calculation request
            clean_request = calculation_request.strip()
            
            # Handle percentage calculations
            if "%" in clean_request or "percent" in clean_request.lower():
                result = self._handle_percentage_calculation(clean_request, all_numbers)
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
    
    def _handle_percentage_calculation(self, request: str, available_numbers: List[Dict[str, Any]]) -> str:
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


class NumericalResponseValidator:
    """Validates numerical responses for accuracy"""
    
    def validate(self, query: str, response: str) -> Dict[str, Any]:
        """Validate a response for numerical accuracy"""
        # Extract numerical answer from response
        numbers = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?(?:\s*%)?)', response)
        
        # Check if calculations are shown
        has_calculations = any(op in response for op in ['+', '-', '*', '/', '='])
        
        # Check if units are included
        has_units = any(unit in response.lower() for unit in 
                        ['%', 'percent', 'dollar', '$', 'million', 'billion', 'euro', '€'])
        
        # Validation result
        validation = {
            "contains_numbers": len(numbers) > 0,
            "shows_calculations": has_calculations,
            "includes_units": has_units,
            "extracted_numbers": numbers,
            "is_valid": len(numbers) > 0 and has_calculations
        }
        
        return {
            "query": query,
            "response": response,
            "validation": validation
        }


def convert_convfinqa_to_documents(data_path: str) -> List[Document]:
    """Convert ConvFinQA dataset to LangChain documents"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    
    for item in data:
        # Process text content
        pre_text = "\n".join(item.get('pre_text', []))
        post_text = "\n".join(item.get('post_text', []))
        
        # Process table if present
        table_content = ""
        if 'table' in item and item['table']:
            # Convert table to markdown format
            table_content = "Table:\n"
            for row in item['table']:
                table_content += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        # Combine all content
        content = f"{pre_text}\n\n{table_content}\n\n{post_text}"
        
        # Create document with metadata
        metadata = {
            "source": item.get('filename', 'unknown'),
            "has_table": bool(table_content),
            "id": item.get('id', '')
        }
        
        # Create LangChain document
        doc = Document(page_content=content, metadata=metadata)
        
        documents.append(doc)
    
    return documents


def create_qa_pairs(data_path: str) -> List[Dict[str, str]]:
    """Extract QA pairs from ConvFinQA dataset for evaluation"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    for item in data:
        if 'qa' in item:
            qa_pairs.append({
                "question": item['qa']['question'],
                "answer": item['qa']['answer'],
                "document_id": item.get('id', '')
            })
        elif 'qa_0' in item and 'qa_1' in item:
            # Handle multiple QA pairs
            for qa_key in ['qa_0', 'qa_1']:
                qa_pairs.append({
                    "question": item[qa_key]['question'],
                    "answer": item[qa_key]['answer'],
                    "document_id": item.get('id', '')
                })
    
    return qa_pairs


def setup_pinecone_vectorstore(documents: List[Document]) -> Pinecone:
    """Set up Pinecone vector store with documents"""
    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX", "convfinqa")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    print(f"Connecting to existing Pinecone index: {pinecone_index_name}")
    
    # Initialize Pinecone with the direct approach that works
    # Use the approach that works for the user
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    print(f"Successfully connected to Pinecone index: {pinecone_index_name}")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create Pinecone vector store
    vectorstore = Pinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    
    return vectorstore


def create_numerical_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for numerical reasoning"""
    template = """You are a financial analyst assistant that specializes in numerical reasoning.
    
CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

Please solve this step by step:
1. Identify the numerical values needed for the calculation
2. Determine the mathematical operations required
3. Perform the calculation showing your work
4. Verify the result makes sense in the context
5. Provide the final answer with appropriate units

Your answer should be precise and include the exact numerical values.
"""
    
    return ChatPromptTemplate.from_template(template)


def main():
    # Load API keys from environment variables or .env file
    load_dotenv()
    
    # Ensure API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Load and process ConvFinQA dataset
    print("Loading and processing ConvFinQA dataset...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dev.json")
    
    # Check if data file exists, if not, print a message
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please download the ConvFinQA dataset and place it in the data directory")
        print("You can download it from: https://github.com/czyssrs/ConvFinQA")
        return
    
    # Load documents and QA pairs
    documents = convert_convfinqa_to_documents(data_path)
    qa_pairs = create_qa_pairs(data_path)
    
    # Process documents
    print("Processing documents...")
    text_splitter = NumericalTextSplitter(chunk_size=512, chunk_overlap=128)
    split_docs = text_splitter.split_documents(documents)
    
    # Process tables
    table_parser = TableParser()
    processed_docs = table_parser.process_documents(split_docs)
    
    # Set up vector store
    print("Setting up vector store...")
    try:
        vectorstore = setup_pinecone_vectorstore(processed_docs)
        print("Successfully set up Pinecone vector store")
    except Exception as e:
        print(f"Error setting up Pinecone: {e}")
        print("Falling back to FAISS vector store...")
        
        # Use FAISS as a fallback
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(processed_docs, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Set up compression retriever for better results
    # Use the same embeddings instance
    embeddings = OpenAIEmbeddings()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )
    
    # Set up LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Create prompt template
    prompt = create_numerical_prompt_template()
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Create calculator tool
    calculator = CalculatorTool()
    
    # Create validator
    validator = NumericalResponseValidator()
    
    # Example query
    print("\nRunning example query...")
    query = "What was the percentage increase in revenue from 2021 to 2022?"
    
    # Get documents for the query
    retrieved_docs = compression_retriever.get_relevant_documents(query)
    
    # Run the query
    result = qa_chain.run(query)
    
    # Validate the result
    validation = validator.validate(query, result)
    
    print("\nQuery:", query)
    print("\nResponse:", result)
    print("\nValidation:", validation["validation"])
    
    # Evaluate on a few examples from the dataset
    print("\nEvaluating on sample queries...")
    sample_queries = qa_pairs[:5]  # Take first 5 QA pairs for demonstration
    
    for i, qa_pair in enumerate(sample_queries):
        query = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        # Run the query
        result = qa_chain.run(query)
        
        # Validate the result
        validation = validator.validate(query, result)
        
        print(f"\nExample {i+1}:")
        print("Question:", query)
        print("Expected Answer:", expected_answer)
        print("Model Answer:", result)
        print("Validation:", validation["validation"])
        print("-" * 80)


if __name__ == "__main__":
    main()
