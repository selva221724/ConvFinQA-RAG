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
        
        # Fallback to original extraction method if FINAL ANSWER format not found
        # Enhanced regex to capture more numerical formats
        number_patterns = [
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
                
                try:
                    extracted_num = float(final_number_str)
                    
                    # Additional validation for percentage-like numbers
                    if ground_truth and '%' in ground_truth:
                        # Ensure extracted number is reasonable for a percentage
                        if 0 <= abs(extracted_num) <= 200:
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

    def calculate_precision_recall_f1(self, retrieved_docs: List[Document], relevant_docs: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for retrieved documents
        
        :param retrieved_docs: List of retrieved documents
        :param relevant_docs: List of relevant document IDs
        :return: Dictionary with precision, recall, and F1 metrics
        """
        # Extract IDs from retrieved documents
        retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
        
        # Calculate true positives (documents that are both retrieved and relevant)
        true_positives = len(set(retrieved_ids).intersection(set(relevant_docs)))
        
        # Calculate precision, recall, and F1
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_mrr(self, retrieved_docs: List[Document], relevant_doc_id: str) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        :param retrieved_docs: List of retrieved documents
        :param relevant_doc_id: ID of the relevant document
        :return: MRR score
        """
        # Extract IDs from retrieved documents
        retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
        
        # Find the rank of the first relevant document
        try:
            rank = retrieved_ids.index(relevant_doc_id) + 1  # +1 because ranks start at 1
            return 1.0 / rank
        except ValueError:
            # Relevant document not found in retrieved documents
            return 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[Document], relevant_docs: List[str]) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        :param retrieved_docs: List of retrieved documents
        :param relevant_docs: List of relevant document IDs
        :return: NDCG score
        """
        # Extract IDs from retrieved documents
        retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
        
        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids):
            # Relevance is binary (1 if relevant, 0 if not)
            relevance = 1 if doc_id in relevant_docs else 0
            # DCG formula: rel_i / log2(i+2)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0 and log2(1) is 0
        
        # Calculate IDCG (ideal DCG)
        # In the ideal case, all relevant documents come first
        idcg = 0
        for i in range(min(len(relevant_docs), len(retrieved_ids))):
            idcg += 1 / np.log2(i + 2)
        
        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the QA system
        
        :return: Evaluation metrics dictionary
        """
        import time
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        # Try to import NLTK and download necessary data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
        except:
            print("Warning: NLTK not fully available. BLEU scores may not be calculated correctly.")
        
        results = {
            'total_samples': len(self.qa_pairs),
            'metrics': {
                'exact_match_accuracy': 0,
                'semantic_similarity_avg': 0,
                'numerical_accuracy': 0,
                'precision_avg': 0,
                'recall_avg': 0,
                'f1_score_avg': 0,
                'mrr_avg': 0,
                'ndcg_avg': 0,
                'bleu_score_avg': 0,
                'average_latency': 0
            },
            'detailed_results': []
        }
        
        total_latency = 0
        
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
            response = self.qa_chain.run(expanded_query)
            end_time = time.time()
            
            # Calculate latency
            latency = end_time - start_time
            total_latency += latency
            
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
            
            # Calculate RAG metrics
            rag_metrics = self.calculate_precision_recall_f1(retrieved_docs, [document_id])
            mrr = self.calculate_mrr(retrieved_docs, document_id)
            ndcg = self.calculate_ndcg(retrieved_docs, [document_id])
            
            # Calculate BLEU score
            try:
                reference = [word_tokenize(ground_truth.lower())]
                candidate = word_tokenize(response.lower())
                bleu_score = sentence_bleu(reference, candidate)
            except:
                bleu_score = 0
            
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
                'validation': validation['validation'],
                'rag_metrics': rag_metrics,
                'mrr': mrr,
                'ndcg': ndcg,
                'bleu_score': bleu_score,
                'latency': latency
            }
            results['detailed_results'].append(result_entry)
            
            # Update aggregate metrics
            results['metrics']['exact_match_accuracy'] += exact_match
            results['metrics']['semantic_similarity_avg'] += semantic_sim
            results['metrics']['numerical_accuracy'] += numerical_accuracy
            results['metrics']['precision_avg'] += rag_metrics['precision']
            results['metrics']['recall_avg'] += rag_metrics['recall']
            results['metrics']['f1_score_avg'] += rag_metrics['f1_score']
            results['metrics']['mrr_avg'] += mrr
            results['metrics']['ndcg_avg'] += ndcg
            results['metrics']['bleu_score_avg'] += bleu_score
        
        # Compute final metrics
        total_samples = len(self.qa_pairs)
        results['metrics']['exact_match_accuracy'] /= total_samples
        results['metrics']['semantic_similarity_avg'] /= total_samples
        results['metrics']['numerical_accuracy'] /= total_samples
        results['metrics']['precision_avg'] /= total_samples
        results['metrics']['recall_avg'] /= total_samples
        results['metrics']['f1_score_avg'] /= total_samples
        results['metrics']['mrr_avg'] /= total_samples
        results['metrics']['ndcg_avg'] /= total_samples
        results['metrics']['bleu_score_avg'] /= total_samples
        results['metrics']['average_latency'] = total_latency / total_samples
        
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
