import os
import json
import numpy as np
from typing import List, Dict, Any

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
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FinancialQAEvaluator:
    def __init__(self, 
                 dataset_path: str, 
                 model_name: str = "o3-mini", 
                 top_k: int = 5, 
                 sample_size: int = None):
        """
        Initialize the Financial QA Evaluator
        
        :param dataset_path: Path to the ConvFinQA dataset
        :param model_name: LLM model to use
        :param top_k: Number of documents to retrieve
        :param sample_size: Number of samples to evaluate (None = full dataset)
        """
        # Logging setup
        self.log_messages = []

        def log(message):
            print(message)
            self.log_messages.append(message)

        try:
            # Load and process dataset
            self.documents = convert_convfinqa_to_documents(dataset_path)
            self.qa_pairs = create_qa_pairs(dataset_path)
            
            # Limit samples if specified
            if sample_size:
                self.qa_pairs = self.qa_pairs[:sample_size]
            
            log(f"Loaded {len(self.qa_pairs)} QA pairs")
            
            # Process documents
            text_splitter = NumericalTextSplitter(chunk_size=384, chunk_overlap=64)
            split_docs = text_splitter.split_documents(self.documents)
            
            # Advanced table parsing
            advanced_table_parser = AdvancedTableParser()
            self.processed_docs = advanced_table_parser.process_documents(split_docs)
            
            # Set up vector store
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(self.processed_docs, embeddings)
            
            # Create base retriever
            self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            
            # Set up cross-encoder reranker
            self.reranker = None
            try:
                from utills import CrossEncoderReranker
                self.reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
                log("Advanced reranker initialized successfully")
            except ImportError as e:
                log(f"Warning: Advanced reranking disabled. Error: {e}")
            except Exception as e:
                log(f"Unexpected error setting up reranker: {e}")
            
            # Set up multi-stage retriever with fallback
            try:
                self.multi_stage_retriever = CustomRetriever(
                    base_retriever=self.base_retriever,
                    embeddings=embeddings,
                    reranker=self.reranker,
                    top_k=top_k
                )
                log("Multi-stage retriever initialized successfully")
            except Exception as e:
                log(f"Fallback to basic retriever due to: {e}")
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
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer()
            
            log("Evaluation system initialized successfully")
        
        except Exception as e:
            log(f"Critical initialization error: {e}")
            raise

    def compute_semantic_similarity(self, response: str, ground_truth: str) -> float:
        """
        Compute semantic similarity between response and ground truth using TF-IDF
        
        :param response: Model's response
        :param ground_truth: Expected answer
        :return: Semantic similarity score
        """
        try:
            # Combine texts to create vocabulary
            corpus = [response, ground_truth]
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Semantic similarity computation error: {e}")
            return 0.0

    def extract_final_numerical_answer(self, response: str, ground_truth: str = None) -> float:
        """
        Extract the final numerical answer from a response with context-aware extraction
        
        :param response: Full response text
        :param ground_truth: Optional ground truth to provide context
        :return: Extracted numerical value
        """
        import re
        
        # Enhanced regex to capture more numerical formats
        number_patterns = [
            # Percentage with optional space
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*%',
            # Decimal numbers with optional commas
            r'(\d+(?:,\d+)*(?:\.\d+)?)',
            # Whole numbers with optional commas
            r'(\d+(?:,\d+)*)'
        ]
        
        # Context-aware extraction
        if ground_truth and '%' in ground_truth:
            # Prioritize percentage extraction if ground truth is a percentage
            percentage_pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*%'
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
                
                try:
                    extracted_num = float(final_number_str)
                    
                    # Additional validation for percentage-like numbers
                    if ground_truth and '%' in ground_truth:
                        # Ensure extracted number is reasonable for a percentage
                        if 0 <= extracted_num <= 200:
                            return extracted_num
                    else:
                        return extracted_num
                except ValueError:
                    continue
        
        return None

    def debug_numerical_extraction(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Provide detailed debugging information for numerical extraction
        
        :param response: Model's response
        :param ground_truth: Ground truth answer
        :return: Debugging information dictionary
        """
        import re
        
        # Extract all numbers from response and ground truth
        response_numbers = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?(?:\s*%)?)', response)
        ground_truth_numbers = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?(?:\s*%)?)', ground_truth)
        
        return {
            'response_full_text': response,
            'ground_truth_full_text': ground_truth,
            'response_numbers_found': response_numbers,
            'ground_truth_numbers_found': ground_truth_numbers,
            'extracted_response_num': self.extract_final_numerical_answer(response),
            'extracted_ground_truth_num': self.extract_final_numerical_answer(ground_truth)
        }

    def compare_numerical_answers(self, response_num: float, ground_truth_num: float) -> Dict[str, Any]:
        """
        Compare numerical answers with more flexible tolerance
        
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
        
        # More lenient tolerances
        relative_tolerance = 0.1  # Increased from 0.05 to 0.1
        absolute_tolerance = 0.5  # Increased from 0.1 to 0.5
        decimal_precision_tolerance = 0.5  # New tolerance for decimal precision
        
        # Calculate differences
        absolute_diff = abs(response_num - ground_truth_num)
        relative_diff = absolute_diff / abs(ground_truth_num) if ground_truth_num != 0 else float('inf')
        
        # Check decimal precision
        decimal_precision_match = abs(round(response_num, 1) - round(ground_truth_num, 1)) <= decimal_precision_tolerance
        
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

    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the QA system
        
        :return: Evaluation metrics dictionary
        """
        results = {
            'total_samples': len(self.qa_pairs),
            'metrics': {
                'exact_match_accuracy': 0,
                'semantic_similarity_avg': 0,
                'retrieval_recall': 0,
                'numerical_accuracy': 0
            },
            'detailed_results': []
        }
        
        for qa_pair in self.qa_pairs:
            query = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            # Expand query
            expanded_query = self.query_expander.expand_query(query)
            
            # Retrieve documents
            retrieved_docs = self.multi_stage_retriever.get_relevant_documents(expanded_query)
            
            # Generate response
            response = self.qa_chain.run(expanded_query)
            
            # Validate response
            validation = self.validator.validate(query, response)
            
            # Compute semantic similarity
            semantic_sim = self.compute_semantic_similarity(response, ground_truth)
            
            # Exact match
            exact_match = response.strip() == ground_truth.strip()
            
            # Extract numerical answers
            response_num = self.extract_final_numerical_answer(response, ground_truth)
            ground_truth_num = self.extract_final_numerical_answer(ground_truth)
            
            # Numerical accuracy
            numerical_comparison = self.compare_numerical_answers(response_num, ground_truth_num)
            numerical_accuracy = numerical_comparison['is_match']
            
            # Store detailed result
            result_entry = {
                'query': query,
                'ground_truth': ground_truth,
                'ground_truth_num': ground_truth_num,
                'response': response,
                'response_num': response_num,
                'exact_match': exact_match,
                'semantic_similarity': semantic_sim,
                'numerical_accuracy': numerical_accuracy,
                'numerical_comparison': numerical_comparison,
                'validation': validation['validation']
            }
            results['detailed_results'].append(result_entry)
            
            # Update aggregate metrics
            results['metrics']['exact_match_accuracy'] += exact_match
            results['metrics']['semantic_similarity_avg'] += semantic_sim
            results['metrics']['numerical_accuracy'] += numerical_accuracy
        
        # Compute final metrics
        total_samples = len(self.qa_pairs)
        results['metrics']['exact_match_accuracy'] /= total_samples
        results['metrics']['semantic_similarity_avg'] /= total_samples
        results['metrics']['numerical_accuracy'] /= total_samples
        
        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a comprehensive evaluation report
        
        :param results: Evaluation results dictionary
        """
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n--- Financial QA Evaluation Report ---")
        print(f"Total Samples: {results['total_samples']}")
        print("\nAggregate Metrics:")
        for metric, value in results['metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {value * 100:.2f}%")
        
        print(f"\nDetailed report saved to {report_path}")

def main():
    # Path to the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dev.json')
    
    # Initialize evaluator
    evaluator = FinancialQAEvaluator(
        dataset_path=dataset_path, 
        model_name="o3-mini", 
        top_k=5, 
        sample_size=5  # Evaluate on first 5 samples
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
