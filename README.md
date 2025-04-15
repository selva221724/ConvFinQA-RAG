# ConvFinQA-RAG

A Retrieval-Augmented Generation (RAG) system for financial question answering using the ConvFinQA dataset. This implementation uses LangChain and Pinecone for vector storage.

## Features

- Numerical reasoning capabilities for financial questions
- Table parsing and structured data handling
- Vector storage with Pinecone (with FAISS fallback)
- Contextual compression for better retrieval
- Calculation tools for arithmetic and percentage operations
- Response validation for numerical accuracy

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Configure API keys:
   - Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

3. Download the ConvFinQA dataset:
   - Clone the ConvFinQA repository: `git clone https://github.com/czyssrs/ConvFinQA.git`
   - Copy the dataset files (train.json, dev.json, test.json) to the `data` directory

4. Run the application:
```
python src/main.py
```

## Dashboard

To use the dashboard for dataset analysis:
```
streamlit run app/dashboard.py
```

## Implementation Details

This implementation uses:
- **LangChain**: For document processing, embedding, retrieval, and LLM integration
- **Pinecone**: For vector storage (with FAISS as a local fallback)
- **OpenAI**: For embeddings and text generation
- **Custom Components**: For numerical reasoning, table parsing, and calculation

## Dataset

The project uses the ConvFinQA dataset, which contains financial questions requiring numerical reasoning. The dataset includes:
- Financial text passages
- Tables with numerical data
- Questions requiring calculations
- Ground truth answers

For more information, visit the [ConvFinQA GitHub repository](https://github.com/czyssrs/ConvFinQA).
