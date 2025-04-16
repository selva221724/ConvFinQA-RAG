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
        # Load and process dataset
        self.documents = convert_convfinqa_to_documents(dataset_path)
        self.qa_pairs = create_qa_pairs(dataset_path)
        
        # Limit samples if specified
        if sample_size:
            self.qa_pairs = self.qa_pairs[:sample_size]
        
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
        try:
            self.reranker = CrossEncoderReranker("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)
        except:
            self.reranker = None
        
        # Set up multi-stage retriever
        self.multi_stage_retriever = CustomRetriever(
            base_retriever=self.base_retriever,
            embeddings=embeddings,
            reranker=self.reranker,
            top_k=top_k
        )
        
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
            
            # Numerical accuracy (for percentage change)
            numerical_accuracy = 0
            try:
                response_num = float(response.rstrip('%'))
                ground_truth_num = float(ground_truth.rstrip('%'))
                numerical_accuracy = 1 if abs(response_num - ground_truth_num) < 0.1 else 0
            except:
                numerical_accuracy = 0
            
            # Store detailed result
            result_entry = {
                'query': query,
                'ground_truth': ground_truth,
                'response': response,
                'exact_match': exact_match,
                'semantic_similarity': semantic_sim,
                'numerical_accuracy': numerical_accuracy,
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
        sample_size=5  # Evaluate on first 50 samples
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
