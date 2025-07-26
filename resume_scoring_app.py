import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import docx2txt
import PyPDF2
import re
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import json
from datetime import datetime
import io

# ---------------------------
# PAGE CONFIG & STYLING
# ---------------------------
st.set_page_config(page_title="AI Resume Scoring App", layout="wide", page_icon="📄")

st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stFileUploader,
    .stDataFrame,
    .res-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 3px 3px 12px rgba(255, 255, 255, 0.15);
        color: white;
    }
    .stTextArea textarea {
        color: black;
        background-color: white;
    }
    .res-card strong {
        color: #ffffff;
    }
    .score-highlight {
        color: #00ffff;
        font-weight: bold;
    }
    .keyword-match {
        color: #90EE90;
        font-weight: bold;
    }
    .missing-keyword {
        color: #FFB6C1;
        font-weight: bold;
    }
    .stApp {
        background-color: black;
    }
    h1, h2, h3, .markdown-text-container, .stMarkdown, .stButton>button, label {
        color: white !important;
    }
    .stButton > button {
        background-color: black !important;
        color: white !important;
        border: 1px solid red;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-1cypcdb {
        background-color: #1a1a1a !important;
    }
    .css-1d391kg .stMarkdown, .css-1d391kg label, .css-1d391kg .css-1cpxqw2, 
    .css-1lcbmhc .stMarkdown, .css-1lcbmhc label, .css-1lcbmhc .css-1cpxqw2,
    .sidebar .stMarkdown, .sidebar label, .sidebar .css-1cpxqw2 {
        color: white !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3 {
        color: white !important;
    }
    /* Slider text */
    .css-1d391kg .css-1vencpc, .css-1lcbmhc .css-1vencpc {
        color: white !important;
    }
    /* Checkbox text */
    .css-1d391kg .css-16huue1, .css-1lcbmhc .css-16huue1 {
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        color: white;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 40px;
        opacity: 0.8;
        color: white;
    }
    /* Enhanced sidebar visibility */
    .css-1d391kg, .css-1lcbmhc, .css-1cypcdb, [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white !important;
    }
    /* Force all sidebar text to be white */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# ENHANCED FUNCTIONS
# ---------------------------
def extract_text_from_file(uploaded_file):
    """Enhanced file extraction with better error handling"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
    return ""

def extract_keywords(text, top_n=20):
    """Extract important keywords from text"""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    filtered_words = [word for word in words if len(word) > 3 and word not in 
                     ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said']]
    return Counter(filtered_words).most_common(top_n)

def extract_contact_info(text):
    """Extract contact information from resume"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return {
        'emails': emails,
        'phones': [''.join(phone) for phone in phones]
    }

def calculate_keyword_match(job_keywords, resume_text):
    """Calculate keyword matching percentage"""
    resume_lower = resume_text.lower()
    matched_keywords = []
    missing_keywords = []
    
    for keyword, _ in job_keywords:
        if keyword in resume_lower:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    match_percentage = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    return match_percentage, matched_keywords, missing_keywords

def analyze_experience_level(text):
    """Analyze experience level based on text content"""
    text_lower = text.lower()
    experience_indicators = {
        'senior': ['senior', 'lead', 'principal', 'architect', 'director', 'manager'],
        'mid': ['mid', 'intermediate', '3-5 years', '4-6 years', '2-4 years'],
        'junior': ['junior', 'entry', 'graduate', 'intern', 'trainee', 'fresher']
    }
    
    scores = {}
    for level, keywords in experience_indicators.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[level] = score
    
    return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'unknown'

def create_score_visualization(ranked_resumes):
    """Create visualization for resume scores"""
    names = [name for name, _, _ in ranked_resumes]
    scores = [score for _, _, score in ranked_resumes]
    
    fig = px.bar(
        x=names, 
        y=scores,
        title="Resume Relevance Scores",
        labels={'x': 'Resume Files', 'y': 'Relevance Score'},
        color=scores,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def generate_word_cloud(text):
    """Generate word cloud from text"""
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='black',
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('black')
        return fig
    except:
        return None

def export_results_to_json(results):
    """Export results to JSON format"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    return json.dumps(export_data, indent=2)

# ---------------------------
# SIDEBAR CONFIGURATION
# ---------------------------
st.sidebar.header("⚙️ Configuration")
score_threshold = st.sidebar.slider("Minimum Score Threshold", 0.0, 1.0, 0.1, 0.05)
top_keywords = st.sidebar.slider("Number of Keywords to Extract", 5, 50, 20)
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", value=True)
auto_export = st.sidebar.checkbox("Auto Export Results", value=False)

# ---------------------------
# MAIN APP UI
# ---------------------------
st.title("📄 AI Resume Scoring App")
st.markdown("<span style='color:white;'>Enhanced AI-powered resume ranking with advanced analytics.</span>", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Main Analysis", "📈 Analytics", "⚙️ Settings"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        job_description = st.text_area("📝 Paste Job Description Here", height=250)
        
    with col2:
        uploaded_files = st.file_uploader(
            "📂 Upload Resumes (TXT, PDF, DOCX)", 
            type=["txt", "pdf", "docx"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"✅ {len(uploaded_files)} file(s) uploaded")

    if st.button("🚀 Score Resumes", type="primary"):
        if not job_description.strip():
            st.warning("Please paste a job description.")
        elif not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            with st.spinner("🔄 Processing resumes..."):
                resumes = []
                resume_names = []
                resume_data = []

                # Extract job keywords
                job_keywords = extract_keywords(job_description, top_keywords)
                
                # Process each resume
                for file in uploaded_files:
                    content = extract_text_from_file(file)
                    if content.strip():
                        resumes.append(content)
                        resume_names.append(file.name)
                        
                        # Extract additional data
                        contact_info = extract_contact_info(content)
                        experience = analyze_experience_level(content)
                        keyword_match, matched, missing = calculate_keyword_match(job_keywords, content)
                        
                        resume_data.append({
                            'name': file.name,
                            'content': content,
                            'contact_info': contact_info,
                            'experience_level': experience,
                            'keyword_match_percentage': keyword_match,
                            'matched_keywords': matched,
                            'missing_keywords': missing
                        })

                if resumes:
                    # TF-IDF + Cosine Similarity
                    documents = [job_description] + resumes
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                    tfidf_matrix = vectorizer.fit_transform(documents)

                    job_vec = tfidf_matrix[0]
                    resume_vecs = tfidf_matrix[1:]
                    scores = cosine_similarity(job_vec, resume_vecs)[0]

                    # Combine scores with resume data
                    for i, data in enumerate(resume_data):
                        data['relevance_score'] = scores[i]

                    # Filter by threshold and sort
                    filtered_resumes = [data for data in resume_data if data['relevance_score'] >= score_threshold]
                    ranked = sorted(filtered_resumes, key=lambda x: x['relevance_score'], reverse=True)

                    if ranked:
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Total Resumes", len(resume_data))
                        with col2:
                            st.metric("✅ Above Threshold", len(ranked))
                        with col3:
                            avg_score = sum(data['relevance_score'] for data in ranked) / len(ranked)
                            st.metric("📈 Average Score", f"{avg_score:.3f}")
                        with col4:
                            best_score = ranked[0]['relevance_score'] if ranked else 0
                            st.metric("🏆 Best Score", f"{best_score:.3f}")

                        # Display ranked resumes
                        st.markdown("## 🔍 Ranked Resumes")
                        
                        for i, data in enumerate(ranked):
                            with st.expander(f"#{i+1} ⬆️ {data['name']} — Score: {data['relevance_score']:.3f}"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="res-card">
                                        <strong>Relevance Score:</strong> <span class="score-highlight">{data['relevance_score']:.3f}</span><br>
                                        <strong>Experience Level:</strong> {data['experience_level'].title()}<br>
                                        <strong>Keyword Match:</strong> <span class="keyword-match">{data['keyword_match_percentage']:.1f}%</span><br><br>
                                        <strong>Content Preview:</strong><br>
                                        <pre>{data['content'][:1000]}...</pre>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    if data['contact_info']['emails']:
                                        st.write("📧 Email:", data['contact_info']['emails'][0])
                                    if data['contact_info']['phones']:
                                        st.write("📞 Phone:", data['contact_info']['phones'][0])
                                    
                                    st.write("✅ Matched Keywords:")
                                    for kw in data['matched_keywords'][:5]:
                                        st.write(f"• {kw}")
                                    
                                    if data['missing_keywords']:
                                        st.write("❌ Missing Keywords:")
                                        for kw in data['missing_keywords'][:3]:
                                            st.write(f"• {kw}")

                        # Summary table with enhanced data
                        table_data = []
                        for i, data in enumerate(ranked):
                            table_data.append({
                                "Rank": i+1,
                                "Resume File": data['name'],
                                "Relevance Score": round(data['relevance_score'], 3),
                                "Keyword Match %": round(data['keyword_match_percentage'], 1),
                                "Experience": data['experience_level'].title(),
                                "Has Email": "✅" if data['contact_info']['emails'] else "❌",
                                "Has Phone": "✅" if data['contact_info']['phones'] else "❌"
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.markdown("### 📊 Detailed Score Summary")
                        st.dataframe(df, use_container_width=True)

                        # Store results in session state for other tabs
                        st.session_state.results = ranked
                        st.session_state.job_keywords = job_keywords
                        
                        # Auto export if enabled
                        if auto_export:
                            export_data = export_results_to_json(table_data)
                            st.download_button(
                                "📥 Download Results (JSON)",
                                export_data,
                                f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json"
                            )
                    else:
                        st.warning(f"No resumes meet the minimum threshold of {score_threshold}")

with tab2:
    if 'results' in st.session_state and show_advanced:
        st.header("📈 Advanced Analytics")
        
        # Score distribution chart
        fig = create_score_visualization([(r['name'], r['content'], r['relevance_score']) for r in st.session_state.results])
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Job Description Keywords")
            job_keywords_df = pd.DataFrame(st.session_state.job_keywords, columns=['Keyword', 'Frequency'])
            st.dataframe(job_keywords_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Experience Level Distribution")
            exp_levels = [r['experience_level'] for r in st.session_state.results]
            exp_counts = Counter(exp_levels)
            
            fig_pie = px.pie(
                values=list(exp_counts.values()),
                names=list(exp_counts.keys()),
                title="Experience Level Distribution"
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Word cloud
        st.subheader("☁️ Job Description Word Cloud")
        if 'job_keywords' in st.session_state:
            job_text = ' '.join([kw for kw, _ in st.session_state.job_keywords])
            wordcloud_fig = generate_word_cloud(job_text)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
    else:
        st.info("Run the analysis first to see advanced analytics.")

with tab3:
    st.header("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Scoring Parameters")
        use_custom_weights = st.checkbox("Use Custom Weights")
        
        if use_custom_weights:
            tfidf_weight = st.slider("TF-IDF Weight", 0.0, 1.0, 0.7)
            keyword_weight = st.slider("Keyword Match Weight", 0.0, 1.0, 0.3)
            st.info("Custom weights will be applied in the next analysis.")
    
    with col2:
        st.subheader("📊 Export Options")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
        include_content = st.checkbox("Include Resume Content", value=False)
        
        if st.button("📥 Export Current Results") and 'results' in st.session_state:
            if export_format == "JSON":
                data = export_results_to_json(st.session_state.results)
                st.download_button("Download JSON", data, "results.json", "application/json")
            elif export_format == "CSV":
                df = pd.DataFrame(st.session_state.results)
                if not include_content:
                    df = df.drop('content', axis=1, errors='ignore')
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown('<div class="footer">Enhanced AI Resume Scoring App | Made with ❤️ using Streamlit | Supports TXT, PDF, DOCX</div>', unsafe_allow_html=True)
