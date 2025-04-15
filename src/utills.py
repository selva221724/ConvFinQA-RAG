import re
from typing import List, Dict, Any, Optional, Tuple
from math import *

# LangChain imports
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings


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
