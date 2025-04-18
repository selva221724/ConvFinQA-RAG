import os
import json
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from tqdm import tqdm

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_pinecone import Pinecone

from utills import (
    AdvancedTableParser, 
    QueryExpander, 
    CrossEncoderReranker, 
    CustomRetriever
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
                # No tables, process normally
                part_metadata = doc.metadata.copy()
                part_metadata["has_table"] = False
                split_parts = super().split_documents([Document(page_content=content, metadata=part_metadata)])
                enhanced_docs.extend(split_parts)
        
        return enhanced_docs
    
    def _extract_tables(self, text: str) -> List[tuple[bool, str]]:
        """Extract table and non-table sections from text"""
        table_pattern = r"(Table:\n(?:.+\n)+?)(?:\n\n|$)"
        sections = []
        last_end = 0
        
        for match in re.finditer(table_pattern, text):
            start, end = match.span()
            if start > last_end:
                sections.append((False, text[last_end:start]))
            sections.append((True, match.group(1)))
            last_end = end
        
        if last_end < len(text):
            sections.append((False, text[last_end:]))
        
        return sections

def convert_convfinqa_to_documents(data_path: str) -> List[Document]:
    """Convert ConvFinQA dataset to LangChain documents"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        pre_text = item['annotation'].get('amt_pre_text', '') 
        post_text = item['annotation'].get('amt_post_text', '')
        table_content = ""
        
        # Process table if present
        table_data = item['annotation'].get('amt_table', None)
        if table_data:
            if isinstance(table_data, str) and ("<table" in table_data or "wikitable" in table_data):
                try:
                    soup = BeautifulSoup(table_data, 'html.parser')
                    table = soup.find('table')
                    if table:
                        rows = table.find_all('tr')
                        table_content = "Table:\n"
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            row_content = "| " + " | ".join([cell.get_text().strip() for cell in cells]) + " |"
                            table_content += row_content + "\n"
                except Exception:
                    # Fallback to simpler approach
                    table_text = table_data
                    table_text = table_text.replace('<table class=\'wikitable\'>', 'Table:')
                    table_text = re.sub(r'<[^>]+>', '', table_text)
                    table_text = re.sub(r'\s+\|\s+', ' | ', table_text)
                    table_content = table_text
            else:
                # Handle list format tables
                table_rows = table_data if isinstance(table_data, list) else []
                table_content = "Table:\n"
                for row in table_rows[:10]:  # Limit to first 10 rows
                    row_cells = [str(cell)[:50] for cell in row]  # Limit cell length
                    table_content += "| " + " | ".join(row_cells) + " |\n"
        
        # Create document
        content = f"{pre_text}\n\n{table_content}\n\n{post_text}"
        metadata = {
            "source": item.get('filename', 'unknown'),
            "has_table": bool(table_content),
            "id": item.get('id', '')
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
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
            for qa_key in ['qa_0', 'qa_1']:
                qa_pairs.append({
                    "question": item[qa_key]['question'],
                    "answer": item[qa_key]['answer'],
                    "document_id": item.get('id', '')
                })
    
    return qa_pairs


def setup_pinecone_vectorstore(documents: List[Document], ingest_flag=False) -> Pinecone:
    """Set up Pinecone vector store with documents"""
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX", "convfinqa")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    print(f"Connecting to existing Pinecone index: {pinecone_index_name}")
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    print(f"Successfully connected to Pinecone index: {pinecone_index_name}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Pinecone(index=index, embedding=embeddings, text_key="text")
    
    if ingest_flag:
        batch_size = 300
        total_batches = (len(documents) + batch_size - 1) // batch_size  # Calculate total batches
        for i in tqdm(range(0, len(documents), batch_size), total=total_batches, desc="Uploading batches"):
            batch = documents[i:i+batch_size]
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                print(f"Error uploading batch: {e}")
    
    return vectorstore


def create_reasoning_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for detailed financial reasoning"""
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
"""
    return ChatPromptTemplate.from_template(template)

def create_answer_extraction_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for extracting precise numerical answers or yes/no responses"""
    template = """Based on the following calculation and reasoning, extract the precise answer.

CALCULATION CONTEXT:
{context}

ORIGINAL QUESTION: {question}

First, analyze the question type:
1. If it contains words like "increase", "decrease", "change", "growth", "difference", or similar terms:
   - This ALWAYS requires a PERCENTAGE answer
   - Look for the percentage calculation in the context (usually includes '× 100%')
   - Use the EXACT percentage value from the calculation

2. If it's a yes/no question (e.g., "Did the value increase?", "Was there a decrease?"):
   - Answer with ONLY "yes" or "no" in lowercase

3. If it asks for a specific amount or value:
   - Use the EXACT number from the calculation
   - Do not include currency symbols or units

CRITICAL RULES:
1. Questions about changes or differences MUST be answered with percentages:
   - "by how much did X increase?" → use the exact percentage
   - "what was the change in X?" → use the exact percentage
   - "how much did X grow?" → use the exact percentage

2. Format rules:
   - Use the EXACT numbers from your calculations
   - Do not modify decimal places or add trailing zeros
   - Keep the negative sign (-) if present
   - For yes/no, use lowercase "yes" or "no"

Look at the calculations in the context and use the EXACT value.

FINAL ANSWER:"""
    return ChatPromptTemplate.from_template(template)

def main():
    # Load environment variables
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Load and process dataset
    print("Loading and processing ConvFinQA dataset...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.json")
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please download the ConvFinQA dataset and place it in the data directory")
        return
    
    # Process documents
    documents = convert_convfinqa_to_documents(data_path)
    qa_pairs = create_qa_pairs(data_path)
    
    print("Processing documents...")
    text_splitter = NumericalTextSplitter(chunk_size=384, chunk_overlap=64)
    split_docs = text_splitter.split_documents(documents)
    
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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(processed_docs, embeddings)
    
    # Set up retrieval system
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Setting up cross-encoder reranker...")
    reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
    
    print("Setting up multi-stage retriever...")
    multi_stage_retriever = CustomRetriever(
        base_retriever=base_retriever,
        embeddings=embeddings,
        reranker=reranker,
        top_k=10
    )
    
    print("Initializing query expander...")
    query_expander = QueryExpander(model_name="gpt-4o-mini")
    
    # Set up LLM and chains
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    reasoning_prompt = create_reasoning_prompt_template()
    extraction_prompt = create_answer_extraction_prompt_template()
    
    # Evaluate sample queries
    print("\nEvaluating on sample queries...")
    sample_queries = qa_pairs[:5]
    
    for i, qa_pair in enumerate(sample_queries):
        query = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        # First chain: Get detailed reasoning
        expanded_query = query_expander.expand_query(query)
        reasoning_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=base_retriever,
            chain_type_kwargs={"prompt": reasoning_prompt}
        )
        
        reasoning_result = reasoning_chain.invoke(expanded_query)
        reasoning_text = (reasoning_result.content if hasattr(reasoning_result, 'content')
                        else reasoning_result['result'] if isinstance(reasoning_result, dict)
                        else str(reasoning_result))
        
        # Second chain: Extract precise answer
        extraction_chain = extraction_prompt | llm
        extraction_result = extraction_chain.invoke({
            "context": reasoning_text,
            "question": query
        })
        
        result = (extraction_result.content if hasattr(extraction_result, 'content')
                 else extraction_result['result'] if isinstance(extraction_result, dict)
                 else str(extraction_result))
        
        # Clean up result
        result = result.strip()
        if result.startswith('FINAL ANSWER:'):
            result = result[len('FINAL ANSWER:'):].strip()
        
        print(f"\nExample {i+1}:")
        print("Question:", query)
        print("Expected Answer:", expected_answer)
        print("Extracted Answer:", result)
        print("-" * 80)

if __name__ == "__main__":
    main()
