# ConvFinQA-RAG: Financial Question Answering with RAG

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![LangChain](https://img.shields.io/badge/🦜️_langchain-latest-green)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A LLM-driven prototype that answers questions based on financial documents using Retrieval Augmented Generation (RAG). Built for the Tomoro.ai interview assessment.

## The Assignment

The goal was to build a prototype that can answer questions based on financial documents (texts, tables, figures) using LLM. Key requirements:

- Demonstrate knowledge and experience in LLM/RAG systems
- Show logic and reasoning behind accuracy metrics choice
- Effectively communicate solutions and ideas
- Use the ConvFinQA dataset (train.json) for development
- Produce a report on metrics and findings

Example from the dataset:
```json
{
    "question": "what was the percentage change in the net cash from operating activities from 2008 to 2009",
    "answer": "14.1%"
}
```

## What's This All About? 🤖

So, I built this RAG system that's basically a financial calculator on steroids! It's not your average chatbot - this one actually understands financial documents and can do some pretty neat tricks:

🧮 **Number Crunching:** Handles all sorts of financial calculations, from simple percentages to complex year-over-year changes.

📊 **Table Master:** You know those annoying financial tables? Yeah, it can read and understand those too!

✅ **Yes/No Pro:** Sometimes you just need a straight answer - it can handle those simple yes/no questions too.

📈 **Math Whiz:** Whether it's calculating growth rates or comparing financial metrics, it's got you covered.

## How Does It Work? 🛠️

### 1. Smart Data Handling
Think of it as a financial document expert that knows exactly how to break down complex reports. I built a special `NumericalTextSplitter` that's like a surgeon - it carefully cuts up documents while keeping all the important numbers and tables intact. No more mangled spreadsheets!

### 2. RAG Magic
The system's got some clever tricks up its sleeve:
- Expands queries to catch all the important details
- Uses a multi-stage approach to find the most relevant info
- Works like a math teacher:
  1. First, shows all the reasoning (no skipping steps!)
  2. Then gives you the final answer, nice and clean

### 3. The Secret Sauce: Custom Prompts
**For Reasoning:**
Built a prompt that turns the LLM into a financial analyst - complete with step-by-step calculations, double-checking work, and making sure all those percentage signs are in the right places!

**For Getting Final Answers:**
Created a super strict format checker that makes sure:
- Numbers come out looking right (no messy decimals!)
- Yes/no answers are crystal clear
- Percentages have their % signs
- Negative numbers keep their minus signs
- Everything's rounded properly

### 4. Quality Control
Because when you're dealing with financial numbers, you can't just wing it! Built a whole evaluation system that:
- Checks if answers are exactly right
- Allows for tiny variations (1% or 5% wiggle room)
- Makes sure plus/minus signs are correct
- Uses fancy math (MAE and RMSE) to track how far off we are
- Keeps detailed logs of everything (because debugging is fun! 😅)

## Project Structure

```
ConvFinQA-RAG/
├── src/
│   ├── main.py           # Core RAG implementation
│   ├── evaluate.py       # Evaluation system
│   ├── utills.py         # Helper components
│   └── evaluation_report.json
├── app/
│   └── dashboard.py      # Streamlit visualization
├── data/
│   ├── dev.json         # Development dataset
│   └── train.json       # Training dataset
├── docs/
│   └── setup.md         # Detailed setup guide
└── helpers/
    └── vector_db_ops.ipynb  # Vector DB operations
```

The project follows a modular structure with clear separation of concerns:
- `src/`: Core implementation files
- `app/`: User interface components
- `data/`: Dataset files
- `docs/`: Documentation
- `helpers/`: Development utilities

## Requirements

- Python 3.11.0
- Key dependencies:
  ```
  langchain
  openai
  pinecone-client
  streamlit
  beautifulsoup4
  sentence-transformers (optional, for cross encoder reranking)
  ```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/ConvFinQA-RAG.git
cd ConvFinQA-RAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - PINECONE_API_KEY
# - PINECONE_INDEX (optional, defaults to "convfinqa")

# Run the system
python src/main.py              # Run QA system
python src/evaluate.py          # Run evaluation
streamlit run app/dashboard.py  # Launch dashboard
```


## Performance Analysis

### Current Metrics (train.json Set) (trust me it was working better with dev.json 😔 )
- Numerical Accuracy: 75.00%
- Within 1% Accuracy: 75.00%
- Within 5% Accuracy: 75.00%
- MAE: 6.81
- RMSE: 13.60
- Average Latency: 16.86 seconds

### Metrics Choice Rationale

1. **Accuracy Metrics**
   - **Exact Match**: Strict accuracy for precise answers
   - **Within 1%/5%**: Allows for minor calculation variations
   - **Sign Match**: Ensures directional correctness (increase/decrease)

2. **Error Metrics**
   - **MAE (Mean Absolute Error)**: Average magnitude of errors
   - **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
   - RMSE > MAE indicates presence of some large errors

3. **Why Not Traditional NLP Metrics?**
   - ROUGE/BLEU not suitable for numerical answers
   - F1/Precision/Recall too binary for numerical evaluation
   - Need metrics that capture error magnitude

### Key Insights

1. **Binary Performance Pattern**
   - 75% accuracy across all thresholds (1%, 5%)
   - Suggests model either:
     * Gets answer exactly right (<1% error)
     * Gets it significantly wrong (>5% error)
   - No "partially correct" answers

2. **Error Distribution**
   - MAE of 6.81 shows average error of ~7 percentage points
   - RMSE (13.60) ≈ 2×MAE suggests:
     * Most errors are moderate
     * Few large outlier errors
     * Consistent with binary performance pattern

3. **Latency Analysis**
   - Average: 16.86 seconds
   - Includes:
     * Document retrieval
     * Two-stage reasoning
     * Answer extraction
   - Acceptable for non-real-time applications

## My RAG Journey 🚀

Hey there! Let me tell you about my journey building this financial QA system. Coming from a background in general text RAG builder, this was my first dive into financial data, and what a learning experience it was!

Over four days (about 2 hours each day between coffee breaks), here's how it went down:

**Day 1: The "What Am I Looking At?" Phase 🤔**
I spent the first day just wrapping my head around financial data. Tables everywhere, numbers that needed to make sense, and percentages that had to be calculated just right. Started with `dev.json` because, well, baby steps! Did some EDA dashboard to understand the data. 

**Day 2: The "Now We're Cooking!" Phase 🛠️**
Decided to go with Pinecone for the vector DB - didn't want to waste time setting up local alternatives. Grabbed OpenAI's endpoints because they're reliable and cost-effective. First RAG pipeline was... let's say "numerically challenged" 😅 But after some custom table parsing and chunking magic, things started looking up.

**Day 3: The "Making It Smart" Phase 💡**
Added query expansion (because context is everything), implemented two-stage answer generation, and made sure our tables didn't turn into spaghetti. The vector DB started actually understanding what numbers meant! It was a good to see the cleaned date loaded to the DB. 

**Day 4: The "Truth Time" Phase 📊**
Evaluation day! Considered going with ROUGE/BLEU but realized they're not great for numbers. Went with MAE/RMSE instead - because when you're dealing with percentages, being off by 0.1 or 10 makes a huge difference! Added all the logging because, well, debugging financial calculations is fun (said no one ever).

🚨 **Warning!** 🚨 

I owe a huge shout-out to my AI sidekick, Claude Sonnet 3.7, for assisting me build the repo and debug like a pro! If you find any bugs or errors in my code, just remember: it's all Claude's fault! 😂 , Just kidding, I had my hands on most of core places and cleaned it up finally, Because most of the time, it couldn't understand the numerical reasoning part, vector db ingestion was poor, conflicting packages and dataset part, it was like a fight in a time loop! But seriously, any mistakes are mine—Claude just provided the caffeine! ☕️

![Python](https://img.shields.io/badge/python-3.11-blue.svg)

### Development Decisions

1. Infrastructure Choices:
   - Pinecone: Quick setup, reliable performance
   - OpenAI (o3-mini): Cost-effective, good performance
   - Python 3.11.0: Latest stable version

2. RAG Enhancements:
   - Custom table parsing
   - Query expansion
   - Two-stage answer generation
   - Numerical context preservation

3. Evaluation Strategy:
   - Mixed metrics for different answer types
   - Error magnitude analysis
   - Comprehensive logging

## Interactive Dashboard

The project includes a Streamlit dashboard (`app/dashboard.py`) that provides EDA of the dataset


## Key Features

1. Enhanced Retrieval
   - Query expansion for better context
   - Multi-stage retrieval pipeline
   - Table-aware document processing
   - Intelligent document chunking

2. Numerical Reasoning
   - Custom prompts for calculations
   - Percentage change handling
   - Sign-aware comparison
   - Threshold-based matching
   - Two-stage answer generation

3. Development Tools
   - Streamlit dashboard for data exploration
   - Comprehensive logging system ( only for `evaluate.py`)
   - Detailed error analysis
   - Performance monitoring

## Things I Could've Done Better (With More Time) ⏰

Looking back, there's so much more I wanted to do! Here's what I'd tackle with more time:

**Better Model Performance:**
I'd love to experiment with GPT-4 and really fine-tune those numerical capabilities. The current model sometimes struggles with complex calculations - would be great to make it more reliable. Also, that reranking feature? Yeah, it needs some love to work properly, When the vector database becomes extensive, retrieving the correct context becomes challenging.

**Proper Production Setup:**
Right now it's more of a prototype, but with time I'd set up a proper CI/CD pipeline with GitHub Actions. Imagine automatic testing, deployment to AWS or Azure, maybe even a nice Kubernetes setup if it's part of a bigger system. Would make updates and scaling so much smoother!

**Enhanced Data Processing:**
The chunking methods could be smarter - sometimes tables get split in weird ways. Would love to implement better table preservation techniques and maybe add some fancy numerical entity recognition.

**Robust Testing:**
While it works, it could use more comprehensive testing. Unit tests, integration tests, the whole nine yards. Would help catch those edge cases where financial calculations go wonky.

**Security & Compliance:**
Financial data needs proper handling - would add PII detection, proper GDPR compliance, and better data privacy measures. Also, the API needs proper authentication and rate limiting for production use.

## 👨🏻‍💻 Author Information

- **Author:** Tamil Selvan
- **Email:** selva221724@gmail.com
- **LinkedIn:** [Tamil Selvan](https://www.linkedin.com/in/selva221724/)
- **GitHub:** [selva221724](https://github.com/selva221724)
- **Stack Overflow:** [Tamil Selvan](https://stackoverflow.com/users/10383650/tamil-selvan)


