# AI Resume Scoring App 📄

An enhanced AI-powered resume ranking application with advanced analytics, built with Streamlit.

## Features

🔍 **Smart Resume Analysis**
- TF-IDF + Cosine Similarity scoring
- Keyword matching with percentage analysis
- Contact information extraction (emails, phones)
- Experience level detection (junior/mid/senior)

📊 **Advanced Analytics**
- Interactive score visualizations with Plotly
- Word cloud generation for job descriptions
- Experience level distribution charts
- Comprehensive keyword analysis

⚙️ **Customizable Settings**
- Adjustable score thresholds
- Configurable keyword extraction
- Export options (JSON, CSV)
- Auto-export functionality

🎨 **Professional UI**
- Dark theme with enhanced visibility
- Tabbed interface for better organization
- Responsive design
- Real-time processing feedback

## Supported File Formats
- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd ai-resume-scoring-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run resume_scoring_app.py
```

## Usage

1. **Paste Job Description**: Enter the job requirements in the text area
2. **Upload Resumes**: Upload multiple resume files (PDF, DOCX, TXT)
3. **Configure Settings**: Adjust score threshold and keyword settings in sidebar
4. **Analyze**: Click "🚀 Score Resumes" to start analysis
5. **Review Results**: Explore ranked results across three tabs:
   - Main Analysis: Detailed scoring and ranking
   - Analytics: Interactive charts and visualizations
   - Settings: Advanced configuration options

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Option 2: Heroku
```bash
pip install gunicorn
echo "web: streamlit run resume_scoring_app.py --server.port=$PORT" > Procfile
```

### Option 3: Railway
1. Connect GitHub repository to Railway
2. Deploy automatically

## Technologies Used

- **Streamlit**: Web app framework
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **Plotly**: Interactive visualizations
- **WordCloud**: Text visualization
- **PyPDF2**: PDF text extraction
- **docx2txt**: Word document processing
- **Pandas**: Data manipulation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Author

Built with ❤️ for efficient resume screening and candidate evaluation.
