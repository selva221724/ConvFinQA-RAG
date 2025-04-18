import os
import json
import numpy as np
from typing import List, Dict, Any
import re
import logging
import time
from tqdm import tqdm


# Import modules from main.py and utills.py
from main import (
    convert_convfinqa_to_documents,
    create_qa_pairs,
    NumericalTextSplitter,
    NumericalResponseValidator
)
from utills import (
    AdvancedTableParser, 
    QueryExpander, 
    CrossEncoderReranker, 
    CustomRetriever,
    AdvancedCalculatorTool
)

# LangChain and OpenAI imports
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone
import pinecone

import ssl

# Bypass SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # For newer Python versions
    ssl._create_default_https_context = _create_unverified_https_context

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinancialQAEvaluator")

class FinancialQAEvaluator:
    def __init__(self, 
                 dataset_path: str, 
                 model_name: str = "o3-mini", 
                 top_k: int = 10, 
                 sample_size: int = None):
        """
        Initialize the Financial QA Evaluator
        
        :param dataset_path: Path to the ConvFinQA dataset
        :param model_name: LLM model to use
        :param top_k: Number of documents to retrieve
        :param sample_size: Number of samples to evaluate (None = full dataset)
        """
        try:
            logger.info(f"Initializing evaluator with model: {model_name}, top_k: {top_k}")
            
            # Load and process dataset
            self.documents = convert_convfinqa_to_documents(dataset_path)
            self.qa_pairs = create_qa_pairs(dataset_path)
            
            # Limit samples if specified
            if sample_size:
                self.qa_pairs = self.qa_pairs[:sample_size]
            
            logger.info(f"Loaded {len(self.qa_pairs)} QA pairs")
            
            # Process documents
            text_splitter = NumericalTextSplitter(chunk_size=384, chunk_overlap=64)
            split_docs = text_splitter.split_documents(self.documents)
            
            # Advanced table parsing
            advanced_table_parser = AdvancedTableParser()
            self.processed_docs = advanced_table_parser.process_documents(split_docs)
            
            # Set up vector store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            pinecone_index_name = os.environ.get("PINECONE_INDEX", "convfinqa")
            
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            logger.info(f"Connecting to existing Pinecone index: {pinecone_index_name}")
            
            # Initialize Pinecone with the direct approach that works
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Create Pinecone vector store
            self.vectorstore = Pinecone(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            
            # Create base retriever
            self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            
            # Set up cross-encoder reranker
            self.reranker = None
            try:
                self.reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
                logger.info("Advanced reranker initialized successfully")
            except ImportError as e:
                logger.warning(f"Advanced reranking disabled. Error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error setting up reranker: {e}")
            
            # Set up multi-stage retriever with fallback
            try:
                self.multi_stage_retriever = CustomRetriever(
                    base_retriever=self.base_retriever,
                    embeddings=embeddings,
                    reranker=self.reranker,
                    top_k=top_k
                )
                logger.info("Multi-stage retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Fallback to basic retriever due to: {e}")
                self.multi_stage_retriever = self.base_retriever
            
            # Initialize query expander
            self.query_expander = QueryExpander(model_name=model_name)
            
            # Set up LLM
            self.llm = ChatOpenAI(
                model_name=model_name
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.base_retriever
            )
            
            # Initialize response validator
            self.validator = NumericalResponseValidator()
            
            logger.info("Evaluation system initialized successfully")
        
        except Exception as e:
            logger.error(f"Critical initialization error: {e}", exc_info=True)
            raise

    def extract_final_numerical_answer(self, response: str, ground_truth: str = None) -> float:
        """
        Extract the final numerical answer from a response with context-aware extraction
        
        :param response: Full response text
        :param ground_truth: Optional ground truth to provide context
        :return: Extracted numerical value
        """
        # First check for "FINAL ANSWER:" format (from chain-of-thought implementation)
        final_answer_match = re.search(r'FINAL ANSWER:\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)', response)
        
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            # Clean the extracted answer
            final_answer = final_answer.replace(',', '')
            # Handle percentage
            if '%' in final_answer:
                final_answer = final_answer.replace('%', '')
                try:
                    return float(final_answer)
                except ValueError:
                    pass
            # Handle currency
            if '$' in final_answer:
                final_answer = final_answer.replace('$', '')
                try:
                    return float(final_answer)
                except ValueError:
                    pass
            # Handle plain number
            try:
                return float(final_answer)
            except ValueError:
                pass
        
        # Enhanced regex patterns to capture more numerical formats
        number_patterns = [
            # Look for specific calculation result patterns
            r'(?:approximately|about|roughly|≈|~)\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)',
            # Look for "is X%" or "was X%" patterns
            r'(?:is|was|of)\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)',
            # Look for result statements
            r'(?:result|answer|calculation|equals|equal to)\s*(?:is|was)?\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)',
            # Look for percentage increase/decrease patterns
            r'(?:increased|decreased|changed|grew|declined|rose|fell)(?:\s+by)?\s+([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)',
            # Look for ratio patterns
            r'(?:ratio|proportion|factor)\s+(?:of|is|was|equals|equal to)?\s*([-+]?[\$]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?(?:\s*:\s*1)?)',
            # Percentage with optional space
            r'([-+]?\d+(?:,\d+)*(?:\.\d+)?)\s*%',
            # Decimal numbers with optional commas
            r'([-+]?\d+(?:,\d+)*(?:\.\d+)?)',
            # Whole numbers with optional commas
            r'([-+]?\d+(?:,\d+)*)'
        ]
        
        # Context-aware extraction
        if ground_truth and '%' in ground_truth:
            # Prioritize percentage extraction if ground truth is a percentage
            percentage_pattern = r'([-+]?\d+(?:,\d+)*(?:\.\d+)?)\s*%'
            numbers = re.findall(percentage_pattern, response)
            
            if numbers:
                final_number_str = numbers[-1].replace(',', '')
                try:
                    return float(final_number_str)
                except ValueError:
                    pass
        
        # Try each pattern in order
        for pattern in number_patterns:
            numbers = re.findall(pattern, response)
            
            if numbers:
                # Take the last number, remove commas, and convert to float
                final_number_str = numbers[-1].replace(',', '')
                
                # Remove currency symbols and percentage signs
                final_number_str = final_number_str.replace('$', '').replace('%', '')
                
                # Handle ratio format (e.g., "4:1")
                if ':' in final_number_str:
                    parts = final_number_str.split(':')
                    if len(parts) == 2 and parts[1] == '1':
                        final_number_str = parts[0]
                
                try:
                    extracted_num = float(final_number_str)
                    
                    # Additional validation for percentage-like numbers
                    if ground_truth and '%' in ground_truth and not ('-' in ground_truth and '-' in final_number_str):
                        # If ground truth is negative percentage but extracted isn't, make it negative
                        if '-' in ground_truth and not '-' in final_number_str:
                            extracted_num = -extracted_num
                    
                    return extracted_num
                except ValueError:
                    continue
        
        return None

    def compare_numerical_answers(self, response_num: float, ground_truth_num: float) -> Dict[str, Any]:
        """
        Compare numerical answers with flexible tolerance
        
        :param response_num: Numerical answer from model
        :param ground_truth_num: Ground truth numerical answer
        :return: Detailed comparison dictionary
        """
        if response_num is None or ground_truth_num is None:
            return {
                'is_match': False,
                'reason': 'Missing numerical value',
                'response_num': response_num,
                'ground_truth_num': ground_truth_num
            }
        
        # Tolerances for different value ranges
        if abs(ground_truth_num) < 1:
            relative_tolerance = 0.2  # 20% for small numbers
            absolute_tolerance = 0.1  # 0.1 absolute difference for small numbers
        elif abs(ground_truth_num) < 10:
            relative_tolerance = 0.15  # 15% for medium numbers
            absolute_tolerance = 0.5  # 0.5 absolute difference for medium numbers
        else:
            relative_tolerance = 0.1  # 10% for large numbers
            absolute_tolerance = 1.0  # 1.0 absolute difference for large numbers
        
        # Special case for percentages
        if abs(ground_truth_num) <= 100 and abs(response_num) <= 100:
            # For percentage values, be more lenient
            decimal_precision_tolerance = 1.0  # Allow 1 percentage point difference
        else:
            decimal_precision_tolerance = abs(ground_truth_num) * 0.05  # 5% of the ground truth value
        
        # Handle sign differences for percentages
        # If one is negative and one is positive, but their absolute values are close
        if (ground_truth_num < 0 and response_num > 0) or (ground_truth_num > 0 and response_num < 0):
            # Check if the absolute values are close
            if abs(abs(ground_truth_num) - abs(response_num)) <= absolute_tolerance:
                return {
                    'is_match': False,
                    'reason': 'Sign mismatch but absolute values are close',
                    'response_num': response_num,
                    'ground_truth_num': ground_truth_num
                }
        
        # Calculate differences
        absolute_diff = abs(response_num - ground_truth_num)
        relative_diff = absolute_diff / abs(ground_truth_num) if ground_truth_num != 0 else float('inf')
        
        # Check decimal precision
        decimal_precision_match = absolute_diff <= decimal_precision_tolerance
        
        # Determine match and reason
        is_match = (
            (relative_diff <= relative_tolerance) or 
            (absolute_diff <= absolute_tolerance) or 
            decimal_precision_match
        )
        
        reason = 'Match' if is_match else 'No match'
        if not is_match:
            reason += f' (Relative diff: {relative_diff:.2%}, Absolute diff: {absolute_diff:.2f})'
        
        return {
            'is_match': is_match,
            'reason': reason,
            'response_num': response_num,
            'ground_truth_num': ground_truth_num,
            'relative_diff': relative_diff,
            'absolute_diff': absolute_diff,
            'decimal_precision_match': decimal_precision_match
        }

    def calculate_retrieval_metrics(self, retrieved_docs: List[Document], document_id: str) -> Dict[str, float]:
        """
        Calculate retrieval metrics (precision, recall, F1, MRR, NDCG)
        
        :param retrieved_docs: List of retrieved documents
        :param document_id: ID of the relevant document
        :return: Dictionary with retrieval metrics
        """
        # Extract IDs from retrieved documents
        retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
        
        # Calculate precision, recall, and F1
        relevant_docs = [document_id]
        true_positives = len(set(retrieved_ids).intersection(set(relevant_docs)))
        
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate MRR
        try:
            rank = retrieved_ids.index(document_id) + 1  # +1 because ranks start at 1
            mrr = 1.0 / rank
        except ValueError:
            # Relevant document not found in retrieved documents
            mrr = 0.0
        
        # Calculate NDCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids):
            # Relevance is binary (1 if relevant, 0 if not)
            relevance = 1 if doc_id in relevant_docs else 0
            # DCG formula: rel_i / log2(i+2)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0 and log2(1) is 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0
        for i in range(min(len(relevant_docs), len(retrieved_ids))):
            idcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the QA system
        
        :return: Evaluation metrics dictionary
        """
        results = {
            'total_samples': len(self.qa_pairs),
            'metrics': {
                'numerical_accuracy': 0,
                'numerical_accuracy_with_sign': 0,
                'numerical_accuracy_within_1pct': 0,
                'numerical_accuracy_within_5pct': 0,
                'numerical_accuracy_within_10pct': 0,
                'retrieval_precision_avg': 0,
                'retrieval_recall_avg': 0,
                'retrieval_f1_score_avg': 0,
                'retrieval_mrr_avg': 0,
                'retrieval_ndcg_avg': 0,
                'average_latency': 0
            },
            'detailed_results': []
        }
        
        total_latency = 0
        
        # Create progress bar
        pbar = tqdm(total=len(self.qa_pairs), desc="Evaluating QA pairs")
        
        for qa_pair in self.qa_pairs:
            query = qa_pair['question']
            ground_truth = qa_pair['answer']
            document_id = qa_pair['document_id']
            
            # Expand query
            expanded_query = self.query_expander.expand_query(query)
            
            # Retrieve documents
            start_time = time.time()
            retrieved_docs = self.multi_stage_retriever.get_relevant_documents(expanded_query)
            
            # Generate response
            try:
                full_response = self.qa_chain.invoke(expanded_query)
                # Extract content from the response object if needed
                if hasattr(full_response, 'content'):
                    full_response = full_response.content
                elif isinstance(full_response, dict) and 'result' in full_response:
                    full_response = full_response['result']
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                full_response = "Error generating response"
            
            end_time = time.time()
            
            # Calculate latency
            latency = end_time - start_time
            total_latency += latency
            
            # Extract numerical answers
            response_num = self.extract_final_numerical_answer(full_response, ground_truth)
            ground_truth_num = self.extract_final_numerical_answer(ground_truth)
            
            # Numerical accuracy
            numerical_comparison = self.compare_numerical_answers(response_num, ground_truth_num)
            numerical_accuracy = numerical_comparison['is_match']
            
            # Calculate additional numerical accuracy metrics
            numerical_accuracy_with_sign = False
            numerical_accuracy_within_1pct = False
            numerical_accuracy_within_5pct = False
            numerical_accuracy_within_10pct = False
            
            if response_num is not None and ground_truth_num is not None:
                # Check if signs match
                signs_match = (response_num >= 0 and ground_truth_num >= 0) or (response_num < 0 and ground_truth_num < 0)
                
                # Calculate relative difference for percentage-based metrics
                if ground_truth_num != 0:
                    rel_diff = abs(response_num - ground_truth_num) / abs(ground_truth_num)
                    numerical_accuracy_with_sign = signs_match
                    numerical_accuracy_within_1pct = rel_diff <= 0.01
                    numerical_accuracy_within_5pct = rel_diff <= 0.05
                    numerical_accuracy_within_10pct = rel_diff <= 0.1
            
            # Calculate retrieval metrics
            retrieval_metrics = self.calculate_retrieval_metrics(retrieved_docs, document_id)
            
            # Store detailed result including the full model response
            result_entry = {
                'query': query,
                'ground_truth': ground_truth,
                'ground_truth_num': ground_truth_num,
                'full_response': full_response,  # Include the full model response
                'response_num': response_num,
                'numerical_accuracy': numerical_accuracy,
                'numerical_comparison': numerical_comparison,
                'numerical_accuracy_with_sign': numerical_accuracy_with_sign,
                'numerical_accuracy_within_1pct': numerical_accuracy_within_1pct,
                'numerical_accuracy_within_5pct': numerical_accuracy_within_5pct,
                'numerical_accuracy_within_10pct': numerical_accuracy_within_10pct,
                'retrieval_metrics': retrieval_metrics,
                'latency': latency
            }
            results['detailed_results'].append(result_entry)
            
            # Update aggregate metrics
            results['metrics']['numerical_accuracy'] += numerical_accuracy
            results['metrics']['numerical_accuracy_with_sign'] += numerical_accuracy_with_sign
            results['metrics']['numerical_accuracy_within_1pct'] += numerical_accuracy_within_1pct
            results['metrics']['numerical_accuracy_within_5pct'] += numerical_accuracy_within_5pct
            results['metrics']['numerical_accuracy_within_10pct'] += numerical_accuracy_within_10pct
            results['metrics']['retrieval_precision_avg'] += retrieval_metrics['precision']
            results['metrics']['retrieval_recall_avg'] += retrieval_metrics['recall']
            results['metrics']['retrieval_f1_score_avg'] += retrieval_metrics['f1_score']
            results['metrics']['retrieval_mrr_avg'] += retrieval_metrics['mrr']
            results['metrics']['retrieval_ndcg_avg'] += retrieval_metrics['ndcg']
            
            # Log brief summary of each evaluation
            logger.debug(f"Query: '{query[:50]}...' - Ground truth: {ground_truth_num} - Response: {response_num} - Match: {numerical_accuracy}")
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Compute final metrics
        total_samples = len(self.qa_pairs)
        for metric in results['metrics']:
            if metric != 'average_latency':
                results['metrics'][metric] /= total_samples
        
        results['metrics']['average_latency'] = total_latency / total_samples
        
        logger.info(f"Evaluation completed. Numerical accuracy: {results['metrics']['numerical_accuracy']:.2%}")
        
        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a comprehensive evaluation report
        
        :param results: Evaluation results dictionary
        """
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.json')
        
        # Create a summary version for console output
        summary_results = {
            'total_samples': results['total_samples'],
            'metrics': results['metrics'],
            # Include a few examples but not all detailed results
            'example_results': results['detailed_results'][:3] if results['detailed_results'] else []
        }
        
        # Save full results to file
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info("\n--- Financial QA Evaluation Report ---")
        logger.info(f"Total Samples: {results['total_samples']}")
        logger.info("\nAggregate Metrics:")
        for metric, value in results['metrics'].items():
            if metric == 'average_latency':
                logger.info(f"{metric.replace('_', ' ').title()}: {value:.2f} seconds")
            else:
                logger.info(f"{metric.replace('_', ' ').title()}: {value * 100:.2f}%")
        
        # Log a few example results
        if results['detailed_results']:
            logger.info("\nExample Query Result:")
            example = results['detailed_results'][0]
            logger.info(f"Query: {example['query']}")
            logger.info(f"Ground Truth: {example['ground_truth']} ({example['ground_truth_num']})")
            logger.info(f"Response Number: {example['response_num']}")
            logger.info(f"Match: {example['numerical_accuracy']} ({example['numerical_comparison']['reason']})")
        
        logger.info(f"\nDetailed report saved to {report_path}")

def main():
    # Path to the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dev.json')
    
    # Initialize evaluator
    evaluator = FinancialQAEvaluator(
        dataset_path=dataset_path, 
        model_name="o3-mini", 
        top_k=10, 
        sample_size=5  # Evaluate on first 5 samples
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
