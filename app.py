# app.py
import os
import streamlit as st
import PyPDF2
import docx2txt
import re
import openai

# ---------- Helpers to extract text ----------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        # docx2txt accepts a path or file-like; we save small temp
        with open("temp_doc.docx", "wb") as f:
            f.write(uploaded_file.read())
        text = docx2txt.process("temp_doc.docx") or ""
        os.remove("temp_doc.docx")
        return text
    except Exception as e:
        return ""

def extract_text(uploaded_file):
    if not uploaded_file:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        # file pointer was consumed maybe; ensure start
        try:
            uploaded_file.seek(0)
        except:
            pass
        return extract_text_from_docx(uploaded_file)
    # fallback: read as text
    try:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8")
    except:
        return ""

# ---------- Basic keyword utilities for fallback ----------
def tokenize_keywords(text):
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower())
    words = text.split()
    # remove short words
    words = [w for w in words if len(w) > 3]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # return top keywords
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [it[0] for it in items[:40]]

def similarity_score(job_keywords, cv_keywords):
    if not job_keywords:
        return 0
    match = sum(1 for k in job_keywords if k in cv_keywords)
    return round(100 * match / max(1, len(job_keywords)))

# ---------- OpenAI wrapper (if key present) ----------
def has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))

def ai_compare_and_suggest(job_text, cv_text):
    """
    If OPENAI_API_KEY is set, call OpenAI. Otherwise return fallback results.
    """
    if not job_text.strip() and not cv_text.strip():
        return {"suggestions": "No input provided.", "questions": [], "match_percent": 0}

    job_keywords = tokenize_keywords(job_text)
    cv_keywords = tokenize_keywords(cv_text)
    score = similarity_score(job_keywords, cv_keywords)

    # Fallback simple results (no API key)
    if not has_openai_key():
        # produce simple suggestions
        missing = [k for k in job_keywords if k not in cv_keywords][:12]
        sugg_text = "Keywords to consider adding or emphasizing in your CV:\n" + ", ".join(missing) if missing else "Your CV already contains many keywords from the job description. Consider emphasizing them in bullet points."
        # generate simple interview questions from top job keywords
        questions = [f"Explain your experience with '{k}'." for k in job_keywords[:6]]
        return {"suggestions": sugg_text, "questions": questions, "match_percent": score}

    # --- If we have an OpenAI key, call the API for richer output ---
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change if needed

    system = "You are a helpful assistant that improves CVs to match job descriptions, writes concise suggestions and generates interview questions."
    user_prompt = (
        f"Job description:\n{job_text}\n\nCandidate CV:\n{cv_text}\n\n"
        "1) Give up to 8 concrete suggestions to improve the CV so it matches the job description (like specific bullets, keywords, phrasing). "
        "2) Give a short match percentage (0-100) with a very brief reason. "
        "3) Generate 6 likely interview questions for this role.\n\n"
        "Respond in JSON with keys: suggestions, match_percent, questions (questions is a list).\n"
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.2,
        )
        text = resp["choices"][0]["message"]["content"]
        # Try to be tolerant: if response is JSON-like, try to parse items via regex
        # We'll do simple extraction:
        # Find match percent
        m = re.search(r"(\d{1,3})\s*%?", text)
        match_percent = int(m.group(1)) if m else score
        # Extract suggestions block (first chunk up to "questions" or so)
        # Fallback: return full text as suggestions
        # Extract questions by lines that end with '?'
        questions = re.findall(r"([A-Z][^?]{5,}\?)", text)
        suggestions = text
        return {"suggestions": suggestions, "questions": questions[:10], "match_percent": match_percent}
    except Exception as e:
        # On API error fallback
        missing = [k for k in job_keywords if k not in cv_keywords][:12]
        sugg_text = "Could not contact AI service — fallback suggestions:\n" + ", ".join(missing)
        questions = [f"Explain your experience with '{k}'." for k in job_keywords[:6]]
        return {"suggestions": sugg_text, "questions": questions, "match_percent": score}

def ai_feedback_on_answer(question, answer, job_text, cv_text):
    if not has_openai_key():
        # simple heuristic feedback
        if len(answer.split()) < 8:
            return "Your answer is short. Try to give a specific example, duration, and result. (e.g., 'I used X for Y and achieved Z')."
        if any(word in answer.lower() for word in tokenize_keywords(job_text)[:6]):
            return "Good: you included relevant keywords. Add a measurable result if possible (numbers, outcomes)."
        return "Try to include a concrete example and one measurable outcome (time saved, percent improved, users impacted)."
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        f"Job description:\n{job_text}\n\nCandidate CV:\n{cv_text}\n\n"
        f"Question: {question}\nCandidate's answer: {answer}\n\n"
        "Give a short constructive feedback in 2 sentences: what was good and 1 specific improvement. "
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful interview coach."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Could not contact AI for feedback. Try to include more concrete specifics and numbers."

# ---------- Streamlit UI ----------
st.set_page_config(page_title="CV Optimization Assistant", page_icon="🛠️", layout="centered")
st.title("🛠️ CV Optimization Assistant (Simple Demo)")
st.markdown("Upload a **job description** and your **CV**. The app will suggest improvements, generate interview questions, and let you practice answers.")

col1, col2 = st.columns(2)
with col1:
    job_file = st.file_uploader("Upload job description (PDF or .txt)", type=["pdf", "txt"])
with col2:
    cv_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

job_text = extract_text(job_file) if job_file else ""
cv_text = extract_text(cv_file) if cv_file else ""

if not job_file and not cv_file:
    st.info("Upload sample job description and CV to start. (You can test without an API key.)")

if st.button("Analyze CV"):
    with st.spinner("Analyzing..."):
        result = ai_compare_and_suggest(job_text, cv_text)
    st.subheader("Match score")
    st.metric("Estimated match", f"{result['match_percent']}%")
    st.subheader("Suggestions to improve your CV")
    st.write(result["suggestions"])
    st.subheader("Likely interview questions")
    questions = result.get("questions", [])
    if not questions:
        st.write("No questions generated.")
    else:
        for i, q in enumerate(questions):
            st.write(f"**Q{i+1}.** {q}")

    if questions:
        st.subheader("Practice answers and get feedback")
        for i, q in enumerate(questions):
            ans = st.text_area(f"Your answer to Q{i+1}:", key=f"ans{i}", height=120)
            if st.button(f"Get feedback for Q{i+1}", key=f"fb{i}"):
                with st.spinner("Getting feedback..."):
                    fb = ai_feedback_on_answer(q, ans, job_text, cv_text)
                st.write("**Feedback:**", fb)

st.markdown("---")
st.write("How to get better results: (1) provide a clear job description; (2) upload a detailed CV; (3) for best AI quality set `OPENAI_API_KEY` environment variable before running.")

