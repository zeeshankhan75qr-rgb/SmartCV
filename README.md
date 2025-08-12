# SmartCV
# SmartCV — AI-Powered CV Optimization Assistant

SmartCV is a simple Streamlit app that:
- Lets you upload a **job description** and your **CV**
- Suggests CV improvements to match the job
- Generates likely interview questions
- Lets you answer and get AI feedback

It works:
- **With OpenAI API Key** — high-quality AI suggestions
- **Without OpenAI API Key** — basic keyword-based suggestions

---

## Features
- Upload job description (PDF/TXT)
- Upload CV (PDF/DOCX)
- Keyword extraction and match scoring
- AI-powered suggestions and interview Qs (if API key set)
- Practice answers with AI feedback

---

## Installation (Local)

```bash
# Clone this repository
git clone https://github.com/yourusername/SmartCV.git
cd SmartCV

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

