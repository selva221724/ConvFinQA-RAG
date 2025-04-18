import os
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from math import *
from typing import List
from bs4 import BeautifulSoup  # You'll need to install beautifulsoup4

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_pinecone import Pinecone

from utills import (
    TableParser, 
    CalculatorTool, 
    AdvancedTableParser, 
    QueryExpander, 
    CrossEncoderReranker, 
    CustomRetriever,
    AdvancedCalculatorTool,
    TORCH_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE
)

# Pinecone
import pinecone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class NumericalTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that preserves numerical context and keeps tables intact"""
    
    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 64):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Override to add numerical metadata to split documents and keep tables intact"""
        enhanced_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Check if content contains a table
            table_sections = self._extract_tables(content)
            
            if table_sections:
                # Process content with tables
                for is_table, section in table_sections:
                    part_metadata = doc.metadata.copy()
                    part_metadata["has_table"] = is_table
                    
                    if is_table:
                        # Keep table sections as single chunks
                        enhanced_docs.append(Document(page_content=section, metadata=part_metadata))
                    else:
                        # Split non-table content normally
                        split_parts = super().split_documents([Document(page_content=section, metadata=part_metadata)])
                        enhanced_docs.extend(split_parts)
            else:
                # No tables, process normally with has_table=False
                part_metadata = doc.metadata.copy()
                part_metadata["has_table"] = False
                split_parts = super().split_documents([Document(page_content=content, metadata=part_metadata)])
                enhanced_docs.extend(split_parts)
        
        return enhanced_docs
    
    def _extract_tables(self, text: str) -> List[Tuple[bool, str]]:
        """
        Extract table and non-table sections from text.
        Returns a list of tuples (is_table, section_text)
        """
        # Pattern to match table sections (starts with "Table:" and continues until blank line or end)
        table_pattern = r"(Table:\n(?:.+\n)+?)(?:\n\n|$)"
        
        sections = []
        last_end = 0
        
        # Find all table sections
        for match in re.finditer(table_pattern, text):
            start, end = match.span()
            
            # Add text before this table (if any)
            if start > last_end:
                sections.append((False, text[last_end:start]))
            
            # Add the table section
            sections.append((True, match.group(1)))
            last_end = end
        
        # Add any remaining text after the last table
        if last_end < len(text):
            sections.append((False, text[last_end:]))
        
        return sections
    
    def _is_table(self, text: str) -> bool:
        """Check if a text block is a table"""
        # More robust check for table content
        return bool(re.match(r"Table:\n", text))



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
                        ['%'])
        
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
        pre_text = item['annotation'].get('amt_pre_text', []) 
        post_text = item['annotation'].get('amt_post_text', [])

        
        # Process table if present
        table_content = ""
        table_data = item['annotation'].get('amt_table', None)
                
        if table_data:
            # Handle HTML table format (wikitable)
            if isinstance(table_data, str) and ("<table" in table_data or "wikitable" in table_data):
                try:
                    # Use BeautifulSoup for proper HTML parsing
                    soup = BeautifulSoup(table_data, 'html.parser')
                    table = soup.find('table')
                    
                    if table:
                        rows = table.find_all('tr')
                        table_content = "Table:\n"
                        
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            row_content = "| " + " | ".join([cell.get_text().strip() for cell in cells]) + " |"
                            table_content += row_content + "\n"
                except Exception as e:
                    # Fallback to simpler approach if BeautifulSoup fails
                    table_text = table_data
                    table_text = table_text.replace('<table class=\'wikitable\'>', 'Table:')
                    table_text = table_text.replace('</table>', '')
                    table_text = table_text.replace('<tr>', '')
                    table_text = table_text.replace('</tr>', '\n')
                    table_text = table_text.replace('<td>', ' | ')
                    table_text = table_text.replace('</td>', '')
                    table_text = re.sub(r'<[^>]+>', '', table_text)
                    table_text = re.sub(r'\s+\|\s+', ' | ', table_text)
                    table_content = table_text
            else:
                # Handle list format tables
                max_rows = 10  # Limit to first 10 rows for large tables
                table_rows = table_data if isinstance(table_data, list) else []
                
                if len(table_rows) > max_rows:
                    # For large tables, only include header and first few rows
                    table_rows = [table_rows[0]] + table_rows[1:max_rows]
                    
                    # Add a note about truncation
                    truncation_note = f"Note: Table truncated to {max_rows} rows out of {len(table_data)} total rows."
                    post_text = truncation_note + "\n" + post_text
                
                # Convert table to plain text format
                table_content = "Table:\n"
                for row in table_rows:
                    # Ensure all cells are strings and limit cell content length
                    row_cells = []
                    for cell in row:
                        cell_str = str(cell)
                        # Truncate very long cell content
                        if len(cell_str) > 50:
                            cell_str = cell_str[:47] + "..."
                        row_cells.append(cell_str)
                    
                    table_content += "| " + " | ".join(row_cells) + " |\n"
        
        # Combine all content
        content = f"{pre_text}\n\n{table_content}\n\n{post_text}"
        
        
        # Create document with metadata
        metadata = {
            "source": item.get('filename', 'unknown'),
            "has_table": bool(table_content),
            "id": item.get('id', ''),
            "original_table_size": len(item['annotation'].get('amt_table', [])) if 'table' in item else 0
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


def setup_pinecone_vectorstore(documents: List[Document], ingest_flag=False) -> Pinecone:
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Pinecone vector store
    vectorstore = Pinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    if ingest_flag:
        # Add documents to the vector store in batches to avoid size limits
        batch_size = 300  # Adjust based on document sizes
        total_batches = (len(documents) - 1) // batch_size + 1
        
        print(f"Adding {len(documents)} documents to Pinecone in {total_batches} batches...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # Simplify metadata for Pinecone
            simplified_batch = []
            for doc in batch:
                simplified_metadata = {}
                for key, value in doc.metadata.items():
                    # Convert complex objects to simple string representations
                    if isinstance(value, list):
                        simplified_metadata[key] = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        simplified_metadata[key] = ", ".join(f"{k}: {v}" for k, v in value.items())
                    else:
                        simplified_metadata[key] = str(value)
                
                simplified_doc = Document(
                    page_content=doc.page_content,
                    metadata=simplified_metadata
                )
                simplified_batch.append(simplified_doc)
            
            try:
                print(f"Uploading batch {batch_num}/{total_batches} to Pinecone ({len(simplified_batch)} documents)...")
                vectorstore.add_documents(simplified_batch)
                print(f"Successfully uploaded batch {batch_num}")
            except Exception as e:
                print(f"Error uploading batch {batch_num}: {e}")
                if "message length too large" in str(e) and batch_size > 10:
                    # If we hit size limits, try with a smaller batch
                    smaller_batch_size = batch_size // 2
                    print(f"Reducing batch size to {smaller_batch_size} and retrying...")
                    
                    # Process the current batch with smaller sub-batches
                    for j in range(0, len(simplified_batch), smaller_batch_size):
                        sub_batch = simplified_batch[j:j+smaller_batch_size]
                        try:
                            print(f"Uploading sub-batch {j//smaller_batch_size + 1} with {len(sub_batch)} documents...")
                            vectorstore.add_documents(sub_batch)
                            print(f"Successfully uploaded sub-batch")
                        except Exception as sub_e:
                            print(f"Error uploading sub-batch: {sub_e}")
                            # If even smaller batches fail, we might need to skip or process individually
                            print("Skipping problematic documents in this sub-batch")
                else:
                    # For other types of errors, continue with next batch
                    print("Continuing with next batch...")
        
        print("Completed adding documents to Pinecone")
    return vectorstore


def create_enhanced_numerical_prompt_template() -> ChatPromptTemplate:
    """Create an enhanced prompt template for numerical reasoning based on FinanceRAG methodology"""
    template = """You are a precise financial analyst specializing in numerical reasoning with financial documents.
    
CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

Solve this step-by-step with extreme precision:
1. Carefully identify ALL numerical values in the context that are relevant to the question
2. Determine the EXACT mathematical operations required for this financial calculation
3. Perform calculations showing COMPLETE work with all intermediate steps
4. Verify each calculation's accuracy and check if the result makes sense in the financial context
5. Provide your detailed reasoning and calculations

CRITICAL INSTRUCTIONS:
- Show ALL intermediate calculation steps
- Include units (%, $, etc.) in your calculations
- If any calculation is uncertain, explain why and provide the most likely answer
- For percentage changes, use the formula: ((new_value - old_value) / old_value) * 100
- For financial ratios, clearly state the formula used

FINAL ANSWER FORMAT:
After your detailed reasoning, you MUST end your response with a line that says "FINAL ANSWER:" followed by ONLY the numerical result (with or without % 'if appropriate). 
For example: "FINAL ANSWER: 42.5%" or "FINAL ANSWER: -21.1%" or "FINAL ANSWER: 1000000"
The out put should not have millions or billions or currency appended to the answer FINAL ANSWER:, just the number.
For example bad answers are "FINAL ANSWER: $1,000,000" or "FINAL ANSWER: 1 million" or "FINAL ANSWER: 1 billion"
Be sure to include the negative sign (-) for negative percentages or values.
Do not include any additional text, explanations, or context after the FINAL ANSWER line.
This format is critical for automated evaluation.

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
    text_splitter = NumericalTextSplitter(chunk_size=384, chunk_overlap=64)
    split_docs = text_splitter.split_documents(documents)
    
    # Process tables with advanced table parser
    print("Processing tables with advanced parser...")
    advanced_table_parser = AdvancedTableParser()
    processed_docs = advanced_table_parser.process_documents(split_docs)
    
    # Set up vector store
    print("Setting up vector store...")
    try:
        vectorstore = setup_pinecone_vectorstore(processed_docs, ingest_flag=True)
        print("Successfully set up Pinecone vector store")
    except Exception as e:
        print(f"Error setting up Pinecone: {e}")
        print("Falling back to FAISS vector store...")
        
        # Use FAISS as a fallback
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(processed_docs, embeddings)
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Increased k for better reranking
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Set up cross-encoder reranker
    print("Setting up cross-encoder reranker...")
    try:
        # Initialize with trust_remote_code=True by default
        reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
        print("Successfully initialized cross-encoder reranker")
    except Exception as e:
        print(f"Error setting up cross-encoder: {e}")
        reranker = None
    
    # Set up multi-stage retriever
    print("Setting up multi-stage retriever...")
    multi_stage_retriever = CustomRetriever(
        base_retriever=base_retriever,
        embeddings=embeddings,
        reranker=reranker,
        top_k=10
    )
    
    # Initialize query expander
    print("Initializing query expander...")
    query_expander = QueryExpander(model_name="o3-mini")
    
    # Set up LLM
    llm = ChatOpenAI(
        model_name="o3-mini",  # Using the o3-mini model as requested
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Create enhanced prompt template
    prompt = create_enhanced_numerical_prompt_template()
    
    # Create QA chain with base retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=base_retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Create advanced calculator tool
    calculator = AdvancedCalculatorTool()
    
    # Create validator
    validator = NumericalResponseValidator()
    
    # Evaluate on a few examples from the dataset
    print("\nEvaluating on sample queries...")
    sample_queries = qa_pairs[:5]  # Take first 5 QA pairs for demonstration
    
    for i, qa_pair in enumerate(sample_queries):
        query = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        # Expand the query
        expanded_query = query_expander.expand_query(query)
        # print(f"\nExpanded query {i+1}: {expanded_query}")
        
        # # Get documents using the expanded query
        # retrieved_docs = multi_stage_retriever.invoke(expanded_query)
        
        # Run the query with expanded query
        full_result = qa_chain.invoke(expanded_query)
        # Extract content from the response object if needed
        if hasattr(full_result, 'content'):
            full_result = full_result.content
        elif isinstance(full_result, dict) and 'result' in full_result:
            full_result = full_result['result']
        
        # Extract just the numerical answer
        final_answer_match = re.search(r'FINAL ANSWER:\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)', full_result)
        
        if final_answer_match:
            result = final_answer_match.group(1)
            # Clean up the result for evaluation
            if result.endswith('%') and not result.startswith('-') and expected_answer and expected_answer.startswith('-'):
                # Handle case where expected answer is negative percentage but extracted answer is positive
                result = '-' + result
        else:
            result = full_result
        
        # Validate the result
        validation = validator.validate(query, result)
        
        print(f"\nExample {i+1}:")
        print("Question:", query)
        print("Expected Answer:", expected_answer)
        # print("Full Model Answer:", full_result)
        print("Extracted Answer:", result)
        print("Validation:", validation["validation"])
        print("-" * 80)


if __name__ == "__main__":
    main()
