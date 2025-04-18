import os
import json
import logging
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Import modules from main.py and utills.py
from main import (
    convert_convfinqa_to_documents,
    create_qa_pairs,
    NumericalTextSplitter,
    create_reasoning_prompt_template,
    create_answer_extraction_prompt_template
)
from utills import (
    AdvancedTableParser, 
    QueryExpander, 
    CrossEncoderReranker, 
    CustomRetriever
)

# LangChain imports
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone
import pinecone

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
    def __init__(self, dataset_path: str, model_name: str = "o3-mini", top_k: int = 10, sample_size: int = None):
        """Initialize the Financial QA Evaluator"""
        try:
            logger.info(f"Initializing evaluator with model: {model_name}, top_k: {top_k}")
            
            # Load and process dataset
            self.documents = convert_convfinqa_to_documents(dataset_path)
            self.qa_pairs = create_qa_pairs(dataset_path)[:sample_size] if sample_size else create_qa_pairs(dataset_path)
            logger.info(f"Loaded {len(self.qa_pairs)} QA pairs")
            
            # Process documents
            text_splitter = NumericalTextSplitter(chunk_size=384, chunk_overlap=64)
            split_docs = text_splitter.split_documents(self.documents)
            self.processed_docs = AdvancedTableParser().process_documents(split_docs)
            
            # Set up vector store
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            pinecone_index_name = os.environ.get("PINECONE_INDEX", "convfinqa")
            
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            logger.info(f"Connecting to existing Pinecone index: {pinecone_index_name}")
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            
            # Initialize components
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.vectorstore = Pinecone(index=index, embedding=self.embeddings, text_key="text")
            self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            
            # Set up reranker and retriever
            self.reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
            self.multi_stage_retriever = CustomRetriever(
                base_retriever=self.base_retriever,
                embeddings=self.embeddings,
                reranker=self.reranker,
                top_k=top_k
            )
            
            # Set up LLM components
            self.query_expander = QueryExpander(model_name=model_name)
            self.llm = ChatOpenAI(model_name=model_name)
            self.reasoning_prompt = create_reasoning_prompt_template()
            self.extraction_prompt = create_answer_extraction_prompt_template()
            
            # Create reasoning chain
            self.reasoning_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.base_retriever,
                chain_type_kwargs={"prompt": self.reasoning_prompt}
            )
            
            logger.info("Evaluation system initialized successfully")
        
        except Exception as e:
            logger.error(f"Critical initialization error: {e}", exc_info=True)
            raise

    def extract_answer(self, response: str) -> Dict[str, Any]:
        """Extract the final answer from a response"""
        response = response.strip().lower()
        if response.startswith('final answer:'):
            response = response[len('final answer:'):].strip()
        
        # Handle yes/no answers
        if response in ['yes', 'no']:
            return {'type': 'boolean', 'value': response}
        
        # Handle numerical answers
        response = response.replace(',', '').replace('$', '')
        try:
            if '%' in response:
                return {'type': 'number', 'value': float(response.replace('%', ''))}
            return {'type': 'number', 'value': float(response)}
        except ValueError:
            return {'type': 'unknown', 'value': None}

    def compare_answers(self, response: Dict[str, Any], ground_truth: str) -> Dict[str, Any]:
        """Compare answers with rounding comparison"""
        ground_truth_answer = self.extract_answer(ground_truth)
        
        # Handle type mismatch
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
            
            # Try different rounding levels
            if round(response_num, 1) == round(ground_truth_num, 1):
                return {
                    'is_match': True,
                    'reason': 'Match after rounding to 1 decimal place',
                    'response': response,
                    'ground_truth': ground_truth_answer
                }
            
            if round(response_num) == round(ground_truth_num):
                return {
                    'is_match': True,
                    'reason': 'Match after rounding to whole number',
                    'response': response,
                    'ground_truth': ground_truth_answer
                }
            
            # Check for close matches
            if ground_truth_num != 0:
                percent_diff = abs((response_num - ground_truth_num) / ground_truth_num) * 100
                if percent_diff <= 1:
                    return {
                        'is_match': True,
                        'reason': f'Within 1% difference ({percent_diff:.2f}%)',
                        'response': response,
                        'ground_truth': ground_truth_answer
                    }
        
        return {
            'is_match': False,
            'reason': 'No match',
            'response': response,
            'ground_truth': ground_truth_answer
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the QA system"""
        results = {
            'total_samples': len(self.qa_pairs),
            'metrics': {
                'numerical_accuracy': 0,
                'boolean_accuracy': 0,
                'total_numerical': 0,
                'total_boolean': 0,
                'numerical_accuracy_with_sign': 0,
                'numerical_accuracy_within_1pct': 0,
                'numerical_accuracy_within_5pct': 0,
                'mae': 0,  # Mean Absolute Error
                'rmse': 0,  # Root Mean Square Error
                'average_latency': 0
            },
            'detailed_results': []
        }
        
        total_latency = 0
        pbar = tqdm(total=len(self.qa_pairs), desc="Evaluating QA pairs")
        
        for qa_pair in self.qa_pairs:
            query = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            # Process query
            start_time = time.time()
            expanded_query = self.query_expander.expand_query(query)
            
            try:
                # Get reasoning
                reasoning_result = self.reasoning_chain.invoke(expanded_query)
                reasoning_text = (reasoning_result.content if hasattr(reasoning_result, 'content')
                                else reasoning_result['result'] if isinstance(reasoning_result, dict)
                                else str(reasoning_result))
                
                # Extract answer
                extraction_chain = self.extraction_prompt | self.llm
                extraction_result = extraction_chain.invoke({
                    "context": reasoning_text,
                    "question": query
                })
                
                full_response = (extraction_result.content if hasattr(extraction_result, 'content')
                               else extraction_result['result'] if isinstance(extraction_result, dict)
                               else str(extraction_result))
                
                # Clean up response
                full_response = full_response.strip()
                if full_response.startswith('FINAL ANSWER:'):
                    full_response = full_response[len('FINAL ANSWER:'):].strip()
                
            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                full_response = "Error generating response"
            
            # Calculate metrics
            latency = time.time() - start_time
            total_latency += latency
            
            response_answer = self.extract_answer(full_response)
            comparison_result = self.compare_answers(response_answer, ground_truth)
            is_match = comparison_result['is_match']
            
            # Update metrics based on answer type
            if response_answer['type'] == 'boolean':
                results['metrics']['total_boolean'] += 1
                if is_match:
                    results['metrics']['boolean_accuracy'] += 1
            elif response_answer['type'] == 'number':
                results['metrics']['total_numerical'] += 1
                if is_match:
                    results['metrics']['numerical_accuracy'] += 1
                
                # Calculate additional metrics for numerical answers
                response_num = response_answer['value']
                ground_truth_num = comparison_result['ground_truth']['value']
                
                signs_match = (response_num >= 0 and ground_truth_num >= 0) or (response_num < 0 and ground_truth_num < 0)
                if signs_match:
                    results['metrics']['numerical_accuracy_with_sign'] += 1
                
                if ground_truth_num != 0:
                    rel_diff = abs(response_num - ground_truth_num) / abs(ground_truth_num)
                    if rel_diff <= 0.01:
                        results['metrics']['numerical_accuracy_within_1pct'] += 1
                    if rel_diff <= 0.05:
                        results['metrics']['numerical_accuracy_within_5pct'] += 1
            
            # Calculate error metrics for numerical answers
            if response_answer['type'] == 'number' and comparison_result['ground_truth']['type'] == 'number':
                response_num = response_answer['value']
                ground_truth_num = comparison_result['ground_truth']['value']
                
                # Calculate absolute error
                abs_error = abs(response_num - ground_truth_num)
                squared_error = abs_error * abs_error
                
                results['metrics']['mae'] += abs_error
                results['metrics']['rmse'] += squared_error
            
            # Store result
            results['detailed_results'].append({
                'query': query,
                'ground_truth': ground_truth,
                'response': response_answer,
                'comparison': comparison_result,
                'latency': latency
            })
            
            # Update progress
            pbar.update(1)
            logger.debug(f"Query: '{query[:50]}...' - Type: {response_answer['type']} - Match: {is_match}")
        
        pbar.close()
        
        # Calculate final metrics
        total_samples = len(self.qa_pairs)
        
        # Calculate accuracy metrics
        if results['metrics']['total_numerical'] > 0:
            num_total = results['metrics']['total_numerical']
            results['metrics']['numerical_accuracy'] /= num_total
            results['metrics']['numerical_accuracy_with_sign'] /= num_total
            results['metrics']['numerical_accuracy_within_1pct'] /= num_total
            results['metrics']['numerical_accuracy_within_5pct'] /= num_total
        
        if results['metrics']['total_boolean'] > 0:
            results['metrics']['boolean_accuracy'] /= results['metrics']['total_boolean']
        
        # Calculate MAE and RMSE
        if results['metrics']['total_numerical'] > 0:
            num_total = results['metrics']['total_numerical']
            results['metrics']['mae'] /= num_total  # Average of absolute errors
            results['metrics']['rmse'] = (results['metrics']['rmse'] / num_total) ** 0.5  # Square root of average squared errors
        
        # Calculate average latency
        results['metrics']['average_latency'] = total_latency / total_samples
        
        logger.info(f"Evaluation completed.")
        logger.info(f"Numerical Questions: {results['metrics']['total_numerical']}, "
                   f"Accuracy: {results['metrics']['numerical_accuracy']:.2%}")
        logger.info(f"Boolean Questions: {results['metrics']['total_boolean']}, "
                   f"Accuracy: {results['metrics']['boolean_accuracy']:.2%}")
        
        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive evaluation report"""
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.json')
        
        # Save full results
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print summary
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
        
        # Show example result
        if results['detailed_results']:
            logger.info("\nExample Query Result:")
            example = results['detailed_results'][0]
            logger.info(f"Query: {example['query']}")
            logger.info(f"Ground Truth: {example['ground_truth']}")
            response_value = example['response']['value']
            if '%' in example['ground_truth']:
                response_value = f"{response_value}%"
            logger.info(f"Response: {response_value}")
            logger.info(f"Answer Type: {example['response']['type']}")
            logger.info(f"Match: {example['comparison']['is_match']} ({example['comparison']['reason']})")
        
        logger.info(f"\nDetailed report saved to {report_path}")

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.json')
    evaluator = FinancialQAEvaluator(dataset_path=dataset_path, model_name="gpt-4o-mini", top_k=10, sample_size=5)
    results = evaluator.evaluate()
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
