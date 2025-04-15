import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


st.set_page_config(layout="wide", page_title="ConvFinQA Dataset Analysis")

@st.cache_data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_operations(program):
    """Extract operations from program string"""
    if not program:
        return []
    
    ops = re.findall(r'([a-zA-Z]+)\(', program)
    return ops

def analyze_question_complexity(question):
    """Analyze question complexity based on length and keywords"""
    complexity = 1  # base complexity
    
    # Length-based complexity
    if len(question.split()) > 20:
        complexity += 1
    if len(question.split()) > 30:
        complexity += 1
        
    # Keyword-based complexity
    keywords = ['percentage', 'ratio', 'change', 'increase', 'decrease', 'growth', 'average', 'total', 'difference']
    for keyword in keywords:
        if keyword in question.lower():
            complexity += 0.5
            
    # Multiple questions indicator
    if '?' in question and question.count('?') > 1:
        complexity += 1
        
    return min(5, complexity)  # Cap at 5

def create_question_df(data):
    """Create a dataframe with question analysis"""
    questions = []
    programs = []
    operations = []
    answer_types = []
    complexities = []
    num_steps = []
    
    for item in data:
        if 'qa' in item:
            questions.append(item['qa']['question'])
            programs.append(item['qa']['program'])
            ops = extract_operations(item['qa']['program'])
            operations.append(ops)
            
            # Determine answer type
            answer = item['qa']['answer']
            if isinstance(answer, str) and '%' in answer:
                answer_types.append('percentage')
            elif isinstance(answer, (int, float)) or (isinstance(answer, str) and answer.replace('.', '', 1).isdigit()):
                answer_types.append('numeric')
            else:
                answer_types.append('text')
            
            complexities.append(analyze_question_complexity(item['qa']['question']))
            
            # Count number of steps in the program
            if 'steps' in item['qa']:
                num_steps.append(len(item['qa']['steps']))
            else:
                num_steps.append(0)
        elif 'qa_0' in item and 'qa_1' in item:
            # Handle multiple QA pairs
            for qa_key in ['qa_0', 'qa_1']:
                questions.append(item[qa_key]['question'])
                programs.append(item[qa_key]['program'])
                ops = extract_operations(item[qa_key]['program'])
                operations.append(ops)
                
                # Determine answer type
                answer = item[qa_key]['answer']
                if isinstance(answer, str) and '%' in answer:
                    answer_types.append('percentage')
                elif isinstance(answer, (int, float)) or (isinstance(answer, str) and answer.replace('.', '', 1).isdigit()):
                    answer_types.append('numeric')
                else:
                    answer_types.append('text')
                
                complexities.append(analyze_question_complexity(item[qa_key]['question']))
                
                # Count number of steps in the program
                if 'steps' in item[qa_key]:
                    num_steps.append(len(item[qa_key]['steps']))
                else:
                    num_steps.append(0)
    
    df = pd.DataFrame({
        'question': questions,
        'program': programs,
        'operations': operations,
        'answer_type': answer_types,
        'complexity': complexities,
        'num_steps': num_steps
    })
    
    return df

def create_table_df(data):
    """Create a dataframe with table analysis"""
    table_sizes = []
    table_rows = []
    table_cols = []
    has_table = []
    
    for item in data:
        if 'table' in item and item['table']:
            table_sizes.append(len(item['table']) * len(item['table'][0]) if item['table'] and len(item['table']) > 0 else 0)
            table_rows.append(len(item['table']))
            table_cols.append(len(item['table'][0]) if item['table'] and len(item['table']) > 0 else 0)
            has_table.append(True)
        else:
            table_sizes.append(0)
            table_rows.append(0)
            table_cols.append(0)
            has_table.append(False)
    
    df = pd.DataFrame({
        'has_table': has_table,
        'table_size': table_sizes,
        'table_rows': table_rows,
        'table_cols': table_cols
    })
    
    return df

def create_text_df(data):
    """Create a dataframe with text analysis"""
    pre_text_lengths = []
    post_text_lengths = []
    total_text_lengths = []
    
    for item in data:
        pre_len = sum(len(t) for t in item.get('pre_text', [])) if 'pre_text' in item else 0
        post_len = sum(len(t) for t in item.get('post_text', [])) if 'post_text' in item else 0
        
        pre_text_lengths.append(pre_len)
        post_text_lengths.append(post_len)
        total_text_lengths.append(pre_len + post_len)
    
    df = pd.DataFrame({
        'pre_text_length': pre_text_lengths,
        'post_text_length': post_text_lengths,
        'total_text_length': total_text_lengths
    })
    
    return df

def flatten_operations(operations_list):
    """Flatten list of operation lists"""
    return [op for sublist in operations_list for op in sublist]

def generate_wordcloud_fig(text):
    """Generate a wordcloud figure that can be displayed with plotly"""
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         stopwords=stop_words).generate(text)
    
    # Convert wordcloud to image
    wordcloud_img = wordcloud.to_array()
    
    # Create a plotly figure from the image
    fig = px.imshow(wordcloud_img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

def main():
    st.title("ConvFinQA Dataset Analysis")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Dataset Overview", "Question Analysis", "Operations Analysis", 
         "Table Analysis", "Text Analysis", "Complexity Analysis", 
         "Answer Types", "Word Clouds"]
    )
    
    try:
        data = load_data("train.json")
        st.sidebar.success(f"Loaded {len(data)} items from train.json")
    except Exception as e:
        st.sidebar.error(f"Error loading train.json: {e}")
        st.error("Please upload train.json file to continue")
        uploaded_file = st.file_uploader("Upload train.json", type="json")
        if uploaded_file:
            data = json.load(uploaded_file)
            st.sidebar.success(f"Loaded {len(data)} items from uploaded file")
        else:
            st.stop()
    
    question_df = create_question_df(data)
    table_df = create_table_df(data)
    text_df = create_text_df(data)
    
    # Combine dataframes
    combined_df = pd.concat([question_df, table_df, text_df], axis=1)
    
    if page == "Dataset Overview":
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Statistics")
            st.write(f"Total number of examples: {len(data)}")
            st.write(f"Total number of questions: {len(question_df)}")
            st.write(f"Examples with tables: {sum(table_df['has_table'])}")
            
            # Sample data display
            st.subheader("Sample Data")
            if st.checkbox("Show sample data"):
                sample_idx = st.slider("Select sample index", 0, len(data)-1, 0)
                st.json(data[sample_idx])
        
        with col2:
            st.subheader("Dataset Composition")
            composition_df = pd.DataFrame({
                'Category': ['With Tables', 'Without Tables'],
                'Count': [sum(table_df['has_table']), len(data) - sum(table_df['has_table'])]
            })
            fig = px.pie(composition_df, values='Count', names='Category',
                         title="Examples with/without Tables")
            st.plotly_chart(fig)
            
            # Answer types distribution
            answer_type_counts = question_df['answer_type'].value_counts().reset_index()
            answer_type_counts.columns = ['Answer Type', 'Count']
            fig = px.pie(answer_type_counts, values='Count', names='Answer Type',
                         title="Answer Type Distribution")
            st.plotly_chart(fig)
    
    elif page == "Question Analysis":
        st.header("Question Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Question Length Distribution")
            question_df['question_length'] = question_df['question'].apply(lambda x: len(x.split()))
            fig = px.histogram(question_df, x='question_length', nbins=30, 
                              title="Question Length Distribution (words)")
            fig.update_layout(xaxis_title="Number of Words", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Question Examples")
            length_option = st.selectbox(
                "Select question length category",
                ["Short (< 10 words)", "Medium (10-20 words)", "Long (> 20 words)"]
            )
            
            if length_option == "Short (< 10 words)":
                examples = question_df[question_df['question_length'] < 10]
            elif length_option == "Medium (10-20 words)":
                examples = question_df[(question_df['question_length'] >= 10) & (question_df['question_length'] <= 20)]
            else:
                examples = question_df[question_df['question_length'] > 20]
            
            st.write(f"Found {len(examples)} {length_option} questions")
            if not examples.empty:
                st.table(examples['question'].sample(min(5, len(examples))).reset_index(drop=True))
        
        with col2:
            st.subheader("Question Complexity")
            fig = px.histogram(question_df, x='complexity', nbins=5, 
                              title="Question Complexity Distribution")
            fig.update_layout(xaxis_title="Complexity Score", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Number of Steps")
            fig = px.histogram(question_df, x='num_steps', nbins=10, 
                              title="Number of Steps Distribution")
            fig.update_layout(xaxis_title="Number of Steps", yaxis_title="Frequency")
            st.plotly_chart(fig)
    
    elif page == "Operations Analysis":
        st.header("Operations Analysis")
        
        # Flatten operations list
        all_operations = flatten_operations(question_df['operations'])
        op_counts = Counter(all_operations)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Common Operations")
            
            # Convert to DataFrame for easier plotting
            op_df = pd.DataFrame({
                'operation': list(op_counts.keys()),
                'count': list(op_counts.values())
            }).sort_values('count', ascending=False)
            
            fig = px.bar(op_df.head(10), x='operation', y='count', 
                         title="Top 10 Operations")
            fig.update_layout(xaxis_title="Operation", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Operation Counts")
            st.table(op_df.head(10).reset_index(drop=True))
        
        with col2:
            st.subheader("Operations per Question")
            question_df['ops_count'] = question_df['operations'].apply(len)
            
            fig = px.histogram(question_df, x='ops_count', nbins=10, 
                              title="Operations per Question Distribution")
            fig.update_layout(xaxis_title="Number of Operations", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Operation Co-occurrence")
            # Get top 5 operations
            top_ops = op_df.head(5)['operation'].tolist()
            
            # Create co-occurrence matrix
            co_occur = np.zeros((len(top_ops), len(top_ops)))
            
            for ops in question_df['operations']:
                for i, op1 in enumerate(top_ops):
                    if op1 in ops:
                        for j, op2 in enumerate(top_ops):
                            if op2 in ops:
                                co_occur[i, j] += 1
            
            co_occur_df = pd.DataFrame(co_occur, index=top_ops, columns=top_ops)
            fig = px.imshow(co_occur_df,
                           color_continuous_scale='Viridis',
                           title="Operation Co-occurrence (Top 5 Operations)")
            st.plotly_chart(fig)
    
    elif page == "Table Analysis":
        st.header("Table Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Table Size Distribution")
            
            # Filter out examples without tables
            tables_df = table_df[table_df['has_table']]
            
            fig = px.histogram(tables_df, x='table_size', nbins=30, 
                              title="Table Size Distribution (cells)")
            fig.update_layout(xaxis_title="Number of Cells", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Table Dimensions")
            fig = px.scatter(tables_df, x='table_rows', y='table_cols',
                            title="Table Dimensions (Rows vs Columns)")
            fig.update_layout(xaxis_title="Number of Rows", yaxis_title="Number of Columns")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Table Rows Distribution")
            fig = px.histogram(tables_df, x='table_rows', nbins=20, 
                              title="Table Rows Distribution")
            fig.update_layout(xaxis_title="Number of Rows", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Table Columns Distribution")
            fig = px.histogram(tables_df, x='table_cols', nbins=15, 
                              title="Table Columns Distribution")
            fig.update_layout(xaxis_title="Number of Columns", yaxis_title="Frequency")
            st.plotly_chart(fig)
    
    elif page == "Text Analysis":
        st.header("Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Length Distribution")
            
            fig = px.histogram(text_df, x='total_text_length', nbins=30, 
                              title="Total Text Length Distribution (characters)")
            fig.update_layout(xaxis_title="Number of Characters", yaxis_title="Frequency")
            st.plotly_chart(fig)
            
            st.subheader("Pre vs Post Text Length")
            fig = px.scatter(text_df, x='pre_text_length', y='post_text_length',
                            title="Pre vs Post Text Length")
            fig.update_layout(xaxis_title="Pre-text Length", yaxis_title="Post-text Length")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Text Length Comparison")
            
            # Create data for box plot
            text_len_data = pd.DataFrame({
                'Length': text_df['pre_text_length'].tolist() + text_df['post_text_length'].tolist(),
                'Type': ['Pre-text'] * len(text_df) + ['Post-text'] * len(text_df)
            })
            
            fig = px.box(text_len_data, x='Type', y='Length',
                        title="Pre-text vs Post-text Length Comparison")
            st.plotly_chart(fig)
            
            st.subheader("Text Length Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                'Pre-text': [
                    text_df['pre_text_length'].mean(),
                    text_df['pre_text_length'].median(),
                    text_df['pre_text_length'].min(),
                    text_df['pre_text_length'].max(),
                    text_df['pre_text_length'].std()
                ],
                'Post-text': [
                    text_df['post_text_length'].mean(),
                    text_df['post_text_length'].median(),
                    text_df['post_text_length'].min(),
                    text_df['post_text_length'].max(),
                    text_df['post_text_length'].std()
                ],
                'Total': [
                    text_df['total_text_length'].mean(),
                    text_df['total_text_length'].median(),
                    text_df['total_text_length'].min(),
                    text_df['total_text_length'].max(),
                    text_df['total_text_length'].std()
                ]
            })
            
            st.table(stats_df)
    
    elif page == "Complexity Analysis":
        st.header("Complexity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Complexity vs. Number of Steps")
            
            fig = px.scatter(question_df, x='complexity', y='num_steps',
                           title="Question Complexity vs. Number of Steps")
            fig.update_layout(xaxis_title="Complexity Score", yaxis_title="Number of Steps")
            st.plotly_chart(fig)
            
            st.subheader("Complexity vs. Operations")
            question_df['ops_count'] = question_df['operations'].apply(len)
            
            fig = px.scatter(question_df, x='complexity', y='ops_count',
                           title="Question Complexity vs. Number of Operations")
            fig.update_layout(xaxis_title="Complexity Score", yaxis_title="Number of Operations")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Complexity by Answer Type")
            
            fig = px.box(question_df, x='answer_type', y='complexity',
                       title="Complexity by Answer Type")
            st.plotly_chart(fig)
            
            st.subheader("Complex Questions Examples")
            complexity_threshold = st.slider("Select complexity threshold", 1.0, 5.0, 4.0, 0.5)
            complex_questions = question_df[question_df['complexity'] >= complexity_threshold]
            
            st.write(f"Found {len(complex_questions)} questions with complexity >= {complexity_threshold}")
            if not complex_questions.empty:
                st.table(complex_questions[['question', 'complexity', 'num_steps']].sample(min(5, len(complex_questions))).reset_index(drop=True))
    
    elif page == "Answer Types":
        st.header("Answer Types Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Answer Type Distribution")
            
            answer_counts = question_df['answer_type'].value_counts().reset_index()
            answer_counts.columns = ['Answer Type', 'Count']
            
            fig = px.bar(answer_counts, x='Answer Type', y='Count',
                       title="Answer Type Distribution")
            st.plotly_chart(fig)
            
            st.subheader("Answer Type by Complexity")
            fig = px.box(question_df, x='answer_type', y='complexity',
                       title="Answer Type by Complexity")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Answer Type by Number of Steps")
            fig = px.box(question_df, x='answer_type', y='num_steps',
                       title="Answer Type by Number of Steps")
            st.plotly_chart(fig)
            
            st.subheader("Operations by Answer Type")
            
            # Get top operations for each answer type
            answer_types = question_df['answer_type'].unique()
            top_ops_by_type = {}
            
            for ans_type in answer_types:
                type_ops = flatten_operations(question_df[question_df['answer_type'] == ans_type]['operations'])
                top_ops_by_type[ans_type] = Counter(type_ops).most_common(5)
            
            selected_type = st.selectbox("Select answer type", answer_types)
            
            if selected_type:
                top_ops = top_ops_by_type[selected_type]
                top_ops_df = pd.DataFrame(top_ops, columns=['Operation', 'Count'])
                
                fig = px.bar(top_ops_df, x='Operation', y='Count',
                           title=f"Top Operations for {selected_type} Answers")
                st.plotly_chart(fig)
    
    elif page == "Word Clouds":
        st.header("Word Cloud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Question Word Cloud")
            
            # Combine all questions
            all_questions = " ".join(question_df['question'])
            
            # Generate and display word cloud using plotly
            wordcloud_fig = generate_wordcloud_fig(all_questions)
            st.plotly_chart(wordcloud_fig)
            
            st.subheader("Most Common Question Words")
            # Tokenize and count words
            stop_words = set(stopwords.words('english'))
            words = re.findall(r'\w+', all_questions.lower())
            word_counts = Counter(word for word in words if word not in stop_words and len(word) > 2)
            
            # Convert to dataframe
            word_df = pd.DataFrame({
                'Word': list(word_counts.keys()),
                'Count': list(word_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Create a bar chart for top words
            fig = px.bar(word_df.head(15), x='Word', y='Count',
                       title="Top 15 Words in Questions")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Word Cloud by Answer Type")
            
            selected_ans_type = st.selectbox(
                "Select answer type for word cloud",
                question_df['answer_type'].unique()
            )
            
            if selected_ans_type:
                # Filter questions by answer type
                filtered_questions = " ".join(question_df[question_df['answer_type'] == selected_ans_type]['question'])
                
                # Generate and display word cloud using plotly
                wordcloud_fig = generate_wordcloud_fig(filtered_questions)
                st.plotly_chart(wordcloud_fig)
                
                st.subheader(f"Top Words for {selected_ans_type} Questions")
                # Tokenize and count words
                words = re.findall(r'\w+', filtered_questions.lower())
                word_counts = Counter(word for word in words if word not in stop_words and len(word) > 2)
                
                # Convert to dataframe
                word_df = pd.DataFrame({
                    'Word': list(word_counts.keys()),
                    'Count': list(word_counts.values())
                }).sort_values('Count', ascending=False)
                
                # Create a bar chart for top words by answer type
                fig = px.bar(word_df.head(10), x='Word', y='Count',
                           title=f"Top 10 Words in {selected_ans_type} Questions")
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
