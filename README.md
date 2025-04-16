# ConvFinQA RAG Evaluation Framework

## Overview
This project implements a comprehensive evaluation framework for a Financial Question Answering (QA) system using the ConvFinQA dataset.

## Features
- Multi-stage document retrieval
- Advanced numerical reasoning
- Comprehensive evaluation metrics
- Detailed performance reporting

## Evaluation Metrics
1. **Exact Match Accuracy**
   - Percentage of responses exactly matching ground truth
   - Indicates precise answer reproduction

2. **Semantic Similarity**
   - Measures semantic closeness between model response and ground truth
   - Uses all-MiniLM-L6-v2 embedding model
   - Captures nuanced understanding beyond exact matching

3. **Numerical Accuracy**
   - Specific to financial numerical reasoning
   - Checks numerical precision, especially for percentage changes
   - Allows small tolerance (< 0.1) for floating-point comparisons

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/ConvFinQA-RAG.git
cd ConvFinQA-RAG
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
- Create a `.env` file in the project root
- Add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Running Evaluation

```bash
python src/evaluate.py
```

### Customization Options
- Modify `sample_size` in `main()` to control number of evaluated samples
- Adjust `top_k` to change document retrieval count
- Select different LLM models by changing `model_name`

## Output
- Generates `evaluation_report.json` with detailed metrics
- Prints aggregate performance metrics to console

## Metrics Interpretation
- **Exact Match Accuracy**: Percentage of responses exactly matching ground truth
- **Semantic Similarity**: Closeness of model's response to ground truth (0-1 scale)
- **Numerical Accuracy**: Precision of numerical calculations

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify your license here]

## Contact
[Your contact information]
