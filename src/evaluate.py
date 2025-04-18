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
            
            # Import prompts from main
            from main import create_reasoning_prompt_template, create_answer_extraction_prompt_template
            
            # Initialize the chains with their respective prompts
            self.reasoning_prompt = create_reasoning_prompt_template()
            self.extraction_prompt = create_answer_extraction_prompt_template()
            
            # Create reasoning chain
            self.reasoning_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.base_retriever,
                chain_type_kwargs={"prompt": self.reasoning_prompt}
            )
            
            # Initialize response validator
            self.validator = NumericalResponseValidator()
            
            logger.info("Evaluation system initialized successfully")
        
        except Exception as e:
            logger.error(f"Critical initialization error: {e}", exc_info=True)
            raise

    def extract_answer(self, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Extract the final answer from a response, handling both numerical and yes/no answers
        
        :param response: Full response text
        :param ground_truth: Optional ground truth to provide context
        :return: Dictionary containing answer type and value
        """
        # Clean up the response by removing "FINAL ANSWER:" prefix if present
        response = response.strip().lower()
        if response.startswith('final answer:'):
            response = response[len('final answer:'):].strip()
        
        # Check for yes/no answer
        if response in ['yes', 'no']:
            return {
                'type': 'boolean',
                'value': response
            }
        
        # Clean up numerical response
        response = response.replace(',', '')  # Remove commas
        response = response.replace('$', '')  # Remove dollar signs
        
        # Try to extract a number
        try:
            # If it's a percentage, handle it
            if '%' in response:
                num = float(response.replace('%', ''))
                return {
                    'type': 'number',
                    'value': num
                }
            # Otherwise, try to convert directly to float
            return {
                'type': 'number',
                'value': float(response)
            }
        except ValueError:
            return {
                'type': 'unknown',
                'value': None
            }

    def compare_answers(self, response: Dict[str, Any], ground_truth: str) -> Dict[str, Any]:
        """
        Compare numerical answers with rounding comparison
        
        :param response_num: Numerical answer from model
        :param ground_truth_num: Ground truth numerical answer
        :return: Detailed comparison dictionary
        """
        # Extract ground truth answer
        ground_truth_answer = self.extract_answer(ground_truth)
        
        # If types don't match, it's not correct
        if response['type'] != ground_truth_answer['type']:
            return {
                'is_match': False,
                'reason': 'Answer type mismatch',
                'response': response,
                'ground_truth': ground_truth_answer
            }
        
        # Handle yes/no answers
        if response['type'] == 'boolean':
            is_match = response['value'] == ground_truth_answer['value']
            return {
                'is_match': is_match,
                'reason': 'Exact match' if is_match else 'No match',
                'response': response,
                'ground_truth': ground_truth_answer
            }
        
        # Handle numerical answers
        if response['type'] == 'number':
            response_num = response['value']
            ground_truth_num = ground_truth_answer['value']
            
            # Round both numbers to 1 decimal place for comparison
            response_rounded = round(response_num, 1)
            ground_truth_rounded = round(ground_truth_num, 1)
            
            # Check if they match after rounding
            if response_rounded == ground_truth_rounded:
                return {
                    'is_match': True,
                    'reason': 'Match after rounding to 1 decimal place',
                    'response': response,
                    'ground_truth': ground_truth_answer,
                    'response_rounded': response_rounded,
                    'ground_truth_rounded': ground_truth_rounded
                }
            
            # If they don't match after rounding to 1 decimal, try rounding to whole number
            response_whole = round(response_num)
            ground_truth_whole = round(ground_truth_num)
            
            if response_whole == ground_truth_whole:
                return {
                    'is_match': True,
                    'reason': 'Match after rounding to whole number',
                    'response': response,
                    'ground_truth': ground_truth_answer,
                    'response_rounded': response_whole,
                    'ground_truth_rounded': ground_truth_whole
                }
            
            # Calculate percentage difference for close matches
            if ground_truth_num != 0:
                percent_diff = abs((response_num - ground_truth_num) / ground_truth_num) * 100
                if percent_diff <= 1:  # Within 1% difference
                    return {
                        'is_match': True,
                        'reason': f'Within 1% difference ({percent_diff:.2f}%)',
                        'response': response,
                        'ground_truth': ground_truth_answer,
                        'percent_diff': percent_diff
                    }
        
        return {
            'is_match': False,
            'reason': 'No match',
            'response': response,
            'ground_truth': ground_truth_answer
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
                'boolean_accuracy': 0,  # For yes/no answers
                'total_numerical': 0,   # Count of numerical questions
                'total_boolean': 0,     # Count of yes/no questions
                'numerical_accuracy_with_sign': 0,
                'numerical_accuracy_within_1pct': 0,
                'numerical_accuracy_within_5pct': 0,
                'precision_avg': 0,
                'recall_avg': 0,
                'f1_score_avg': 0,
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
            
            # First chain: Get detailed reasoning
            try:
                reasoning_result = self.reasoning_chain.invoke(expanded_query)
                if hasattr(reasoning_result, 'content'):
                    reasoning_text = reasoning_result.content
                elif isinstance(reasoning_result, dict) and 'result' in reasoning_result:
                    reasoning_text = reasoning_result['result']
                else:
                    reasoning_text = str(reasoning_result)
                
                # Second chain: Extract precise answer
                extraction_chain = self.extraction_prompt | self.llm
                
                # Get precise answer using the reasoning text as context
                extraction_result = extraction_chain.invoke({
                    "context": reasoning_text,
                    "question": query
                })
                
                if hasattr(extraction_result, 'content'):
                    full_response = extraction_result.content
                elif isinstance(extraction_result, dict) and 'result' in extraction_result:
                    full_response = extraction_result['result']
                else:
                    full_response = str(extraction_result)
                
                # Clean up the result by removing "FINAL ANSWER:" prefix if present
                full_response = full_response.strip()
                if full_response.startswith('FINAL ANSWER:'):
                    full_response = full_response[len('FINAL ANSWER:'):].strip()
                
            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                full_response = "Error generating response"
            
            end_time = time.time()
            
            # Calculate latency
            latency = end_time - start_time
            total_latency += latency
            
            # Extract and compare answers
            response_answer = self.extract_answer(full_response)
            comparison_result = self.compare_answers(response_answer, ground_truth)
            
            # Calculate metrics based on answer type
            is_match = comparison_result['is_match']
            
            # Track answer type counts and accuracy
            if response_answer['type'] == 'boolean':
                results['metrics']['total_boolean'] += 1
                if is_match:
                    results['metrics']['boolean_accuracy'] += 1
            elif response_answer['type'] == 'number' and comparison_result['ground_truth']['type'] == 'number':
                results['metrics']['total_numerical'] += 1
                if is_match:
                    results['metrics']['numerical_accuracy'] += 1
                
                # Calculate additional numerical metrics
                response_num = response_answer['value']
                ground_truth_num = comparison_result['ground_truth']['value']
                
                # Check if signs match
                signs_match = (response_num >= 0 and ground_truth_num >= 0) or (response_num < 0 and ground_truth_num < 0)
                numerical_accuracy_with_sign = signs_match
                
                # Calculate relative difference for percentage-based metrics
                if ground_truth_num != 0:
                    rel_diff = abs(response_num - ground_truth_num) / abs(ground_truth_num)
                    numerical_accuracy_within_1pct = rel_diff <= 0.01
                    numerical_accuracy_within_5pct = rel_diff <= 0.05
            
            # Calculate precision, recall, and F1 based on match
            precision = 1.0 if is_match else 0.0
            recall = 1.0 if is_match else 0.0
            f1_score = 1.0 if is_match else 0.0
            
            # Store detailed result
            result_entry = {
                'query': query,
                'ground_truth': ground_truth,
                'full_response': full_response,
                'response': response_answer,
                'comparison': comparison_result,
                'numerical_accuracy': is_match,
                'numerical_accuracy_with_sign': numerical_accuracy_with_sign,
                'numerical_accuracy_within_1pct': numerical_accuracy_within_1pct,
                'numerical_accuracy_within_5pct': numerical_accuracy_within_5pct,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'latency': latency
            }
            results['detailed_results'].append(result_entry)
            
            # Update aggregate metrics
            results['metrics']['numerical_accuracy_with_sign'] += numerical_accuracy_with_sign
            results['metrics']['numerical_accuracy_within_1pct'] += numerical_accuracy_within_1pct
            results['metrics']['numerical_accuracy_within_5pct'] += numerical_accuracy_within_5pct
            results['metrics']['precision_avg'] += precision
            results['metrics']['recall_avg'] += recall
            results['metrics']['f1_score_avg'] += f1_score
            
            # Log brief summary of each evaluation
            logger.debug(f"Query: '{query[:50]}...' - Type: {response_answer['type']} - Ground truth: {ground_truth} - Response: {response_answer['value']} - Match: {is_match}")
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Compute final metrics
        total_samples = len(self.qa_pairs)
        
        # Calculate accuracy percentages
        if results['metrics']['total_numerical'] > 0:
            results['metrics']['numerical_accuracy'] /= results['metrics']['total_numerical']
            results['metrics']['numerical_accuracy_with_sign'] /= results['metrics']['total_numerical']
            results['metrics']['numerical_accuracy_within_1pct'] /= results['metrics']['total_numerical']
            results['metrics']['numerical_accuracy_within_5pct'] /= results['metrics']['total_numerical']
        
        if results['metrics']['total_boolean'] > 0:
            results['metrics']['boolean_accuracy'] /= results['metrics']['total_boolean']
        
        # Calculate other metrics
        for metric in ['precision_avg', 'recall_avg', 'f1_score_avg']:
            results['metrics'][metric] /= total_samples
        
        results['metrics']['average_latency'] = total_latency / total_samples
        
        logger.info(f"Evaluation completed.")
        logger.info(f"Numerical Questions: {results['metrics']['total_numerical']}, Accuracy: {results['metrics']['numerical_accuracy']:.2%}")
        logger.info(f"Boolean Questions: {results['metrics']['total_boolean']}, Accuracy: {results['metrics']['boolean_accuracy']:.2%}")
        
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
            metric_name = metric.replace('_', ' ').title()
            if metric == 'average_latency':
                logger.info(f"{metric_name}: {value:.2f} seconds")
            elif metric in ['total_numerical', 'total_boolean']:
                logger.info(f"{metric_name}: {int(value)}")
            elif metric.startswith(('numerical_', 'boolean_', 'precision_', 'recall_', 'f1_')):
                logger.info(f"{metric_name}: {value * 100:.2f}%")
            else:
                logger.info(f"{metric_name}: {value}")
        
        # Log a few example results
        if results['detailed_results']:
            logger.info("\nExample Query Result:")
            example = results['detailed_results'][0]
            logger.info(f"Query: {example['query']}")
            logger.info(f"Ground Truth: {example['ground_truth']}")
            # Add % sign if ground truth has it
            response_value = example['response']['value']
            if '%' in example['ground_truth']:
                response_value = f"{response_value}%"
            logger.info(f"Response: {response_value}")
            logger.info(f"Answer Type: {example['response']['type']}")
            logger.info(f"Match: {example['numerical_accuracy']} ({example['comparison']['reason']})")
        
        logger.info(f"\nDetailed report saved to {report_path}")

def main():
    # Path to the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dev.json')
    
    # Initialize evaluator
    evaluator = FinancialQAEvaluator(
        dataset_path=dataset_path, 
        model_name="gpt-4o-mini", 
        top_k=10, 
        sample_size=5  # Evaluate on first 5 samples
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
