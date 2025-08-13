# app.py
import os
import streamlit as st
import PyPDF2
import docx2txt
import re
import openai
from datetime import datetime
from fpdf import FPDF

# ----------------- Helpers: files -> text -----------------
def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        uploaded_file.seek(0)
        with open("temp_doc.docx", "wb") as f:
            f.write(uploaded_file.read())
        text = docx2txt.process("temp_doc.docx") or ""
        os.remove("temp_doc.docx")
        return text
    except Exception:
        return ""

def extract_text(uploaded_file):
    if not uploaded_file:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    # try reading as text
    try:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8")
    except Exception:
        return ""

# ----------------- Simple keyword utils (fallback) -----------------
def tokenize_keywords(text):
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", (text or "").lower())
    words = text.split()
    words = [w for w in words if len(w) > 3]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [it[0] for it in items[:40]]

def similarity_score(job_keywords, cv_keywords):
    if not job_keywords:
        return 0
    match = sum(1 for k in job_keywords if k in cv_keywords)
    return round(100 * match / max(1, len(job_keywords)))

# ----------------- OpenAI helpers -----------------
def has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))

def openai_generate_questions_from_job(job_text, n_questions=6):
    # Use LLM when available, otherwise fallback to keyword-constructed questions
    job_text_short = (job_text or "").strip()
    if not job_text_short:
        return []

    if not has_openai_key():
        # fallback: use top keywords to craft concrete questions
        keys = tokenize_keywords(job_text_short)[:8]
        qs = []
        for k in keys[:n_questions]:
            qs.append(f"Describe a project where you used '{k}' and what the result was.")
        return qs

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    system = "You are an assistant that creates interview questions strictly from the provided job description. Produce concise, role-relevant technical and behavioral questions."
    user = f"Job description:\n{job_text_short}\n\nGenerate {n_questions} interview questions that are specifically derived from this job description. Focus on skills, tools, responsibilities and likely scenarios the candidate will face. Return them as a numbered plain list (no JSON)."
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            max_tokens=400,
            temperature=0.2,
        )
        txt = resp["choices"][0]["message"]["content"]
        # extract lines that look like questions or sentences
        lines = [l.strip(" .\n\t") for l in re.split(r"\n+", txt) if l.strip()]
        # prefer lines that end with question mark, else keep sentences
        questions = []
        for line in lines:
            # remove leading numbering like "1. "
            line = re.sub(r"^\s*\d+[\).\s-]*", "", line).strip()
            if len(line) > 10:
                questions.append(line if line.endswith("?") else (line + "?"))
            if len(questions) >= n_questions:
                break
        return questions
    except Exception:
        # fallback if API fails
        keys = tokenize_keywords(job_text_short)[:8]
        return [f"Describe a project where you used '{k}' and what the result was." for k in keys[:n_questions]]

def openai_suggest_improvements(job_text, cv_text):
    # Returns a string of suggestions and a match percent
    job_text_short = (job_text or "").strip()
    cv_text_short = (cv_text or "").strip()
    job_keys = tokenize_keywords(job_text_short)
    cv_keys = tokenize_keywords(cv_text_short)
    base_score = similarity_score(job_keys, cv_keys)

    if not has_openai_key():
        missing = [k for k in job_keys if k not in cv_keys][:12]
        sugg_text = "Fallback suggestions (no API key):\n" + (", ".join(missing) if missing else "Your CV already includes many job keywords. Emphasize results and numbers.")
        return sugg_text, base_score

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    system = "You are a CV coach. Provide concise specific suggestions to improve the CV to match the job description."
    prompt = (
        f"Job description:\n{job_text_short}\n\nCandidate CV:\n{cv_text_short}\n\n"
        "1) Give up to 8 concrete, short suggestions (bullet-like) to improve the CV so it better matches the job description. "
        "2) Give a short match percent (0-100) and one sentence reason.\n\nRespond clearly; the assistant UI will display the whole text."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system}, {"role":"user","content":prompt}],
            max_tokens=600,
            temperature=0.2
        )
        text = resp["choices"][0]["message"]["content"]
        # try to extract percent
        m = re.search(r"(\d{1,3})\s*%?", text)
        match_percent = int(m.group(1)) if m else base_score
        return text, match_percent
    except Exception:
        missing = [k for k in job_keys if k not in cv_keys][:12]
        sugg_text = "Could not contact AI — fallback suggestions:\n" + (", ".join(missing) if missing else "No obvious missing keywords.")
        return sugg_text, base_score

def openai_feedback_on_answer(question, answer, job_text, cv_text):
    if not has_openai_key():
        if not answer or len(answer.split()) < 8:
            return "Short answer. Add a specific example, duration, and result. Example: 'I used X for Y and achieved Z.'"
        # positive comment if keywords present
        if any(k in answer.lower() for k in tokenize_keywords(job_text)[:6]):
            return "Good: you included relevant keywords. Add a measurable result (numbers/outcomes)."
        return "Try to add a concrete example and one measurable outcome (time saved, percent improved)."

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        f"Job description:\n{job_text}\n\nCandidate CV:\n{cv_text}\n\n"
        f"Question: {question}\nCandidate's answer: {answer}\n\n"
        "Give one-sentence praise and one-sentence specific improvement suggestion. Keep it short."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a helpful interview coach."}, {"role":"user","content":prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Could not contact AI for feedback. Try to include more concrete specifics and numbers."

# ----------------- PDF report -----------------
def build_pdf_report(title, job_text, cv_text, suggestions, match_percent, questions, answers, feedbacks, out_path="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, f"Match Score: {match_percent}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Suggestions:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in str(suggestions).split("\n"):
        pdf.multi_cell(0, 6, f"- {line.strip()}")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Interview Questions & Answers:", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, q in enumerate(questions):
        pdf.multi_cell(0, 6, f"Q{i+1}. {q}")
        ans = answers.get(i, "")
        pdf.multi_cell(0, 6, f"A: {ans}")
        fb = feedbacks.get(i, "")
        pdf.multi_cell(0, 6, f"Feedback: {fb}")
        pdf.ln(2)

    pdf.output(out_path)
    return out_path

# ----------------- Streamlit UI and logic -----------------
st.set_page_config(page_title="SmartCV — CV Optimization", page_icon="🛠️", layout="wide")
st.title("🛠️ SmartCV — CV Optimization Assistant")

# Initialize session state keys
if "questions" not in st.session_state:
    st.session_state.questions = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ""
if "match_percent" not in st.session_state:
    st.session_state.match_percent = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}   # map index -> text
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = {}  # map index -> feedback text
if "history" not in st.session_state:
    st.session_state.history = []  # list of past results

# Sidebar: instructions and environment info
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Upload job description (PDF or TXT) and your CV (PDF or DOCX).  
    2. Preview text, then click **Analyze**.  
    3. Practice answers — answers are saved in the session.  
    4. Click **Get feedback** per question.  
    5. Download a PDF report if you want.
    """)
    st.markdown("---")
    st.write("AI key set?" + (" ✅" if has_openai_key() else " ❌ (using fallback)"))
    st.caption("To enable full AI features, set OPENAI_API_KEY in environment (or Streamlit secrets).")

# Layout: left = uploads & preview, right = results & practice
left, right = st.columns([1, 1.3])

with left:
    st.subheader("1) Upload files")
    job_file = st.file_uploader("Job description (PDF or TXT)", type=["pdf","txt"])
    cv_file = st.file_uploader("Your CV (PDF or DOCX)", type=["pdf","docx"])

    # Extract text but don't re-run heavy ops unnecessarily
    job_text = extract_text(job_file) if job_file else (st.session_state.get("last_job_text","") or "")
    cv_text = extract_text(cv_file) if cv_file else (st.session_state.get("last_cv_text","") or "")

    st.subheader("2) Preview extracted text")
    with st.expander("Job description preview", expanded=False):
        st.write(job_text if job_text.strip() else "No job description uploaded yet.")
    with st.expander("CV preview", expanded=False):
        st.write(cv_text if cv_text.strip() else "No CV uploaded yet.")

    # store last texts so re-analyze without re-upload works
    if st.button("Use uploaded texts for analysis (store)"):
        st.session_state["last_job_text"] = job_text
        st.session_state["last_cv_text"] = cv_text
        st.success("Texts stored. Now go to Analyze on the right.")

    st.markdown("---")
    if st.button("Clear session (answers + history)"):
        st.session_state.questions = []
        st.session_state.suggestions = ""
        st.session_state.match_percent = 0
        st.session_state.answers = {}
        st.session_state.feedbacks = {}
        st.session_state.history = []
        st.session_state.last_job_text = ""
        st.session_state.last_cv_text = ""
        st.success("Session cleared.")

with right:
    st.subheader("3) Analysis & Practice")

    # Use stored texts if available (preferred)
    job_text = st.session_state.get("last_job_text", job_text)
    cv_text = st.session_state.get("last_cv_text", cv_text)

    st.write("Job text length:", len(job_text), "chars — CV text length:", len(cv_text), "chars")

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if st.button("Analyze"):
            # generate suggestions and questions
            suggestions, match_percent = openai_suggest_improvements(job_text, cv_text)
            st.session_state.suggestions = suggestions
            st.session_state.match_percent = match_percent
            st.session_state.questions = openai_generate_questions_from_job(job_text, n_questions=6)
            # initialize answers/feedback for each question index
            for i in range(len(st.session_state.questions)):
                st.session_state.answers.setdefault(i, "")
                st.session_state.feedbacks.setdefault(i, "")
            # record history entry
            st.session_state.history.append({
                "ts": datetime.utcnow().isoformat(),
                "match": match_percent,
                "suggestions": suggestions,
                "questions": st.session_state.questions.copy()
            })
            st.success("Analysis complete.")
    with col_b:
        if st.button("Re-analyze using stored texts"):
            suggestions, match_percent = openai_suggest_improvements(job_text, cv_text)
            st.session_state.suggestions = suggestions
            st.session_state.match_percent = match_percent
            st.success("Re-analysis complete.")
    with col_c:
        if st.button("Download PDF report"):
            # ensure we have content
            if not st.session_state.questions and not st.session_state.suggestions:
                st.warning("Run Analyze first to generate content.")
            else:
                path = build_pdf_report(
                    title="SmartCV Report",
                    job_text=job_text,
                    cv_text=cv_text,
                    suggestions=st.session_state.suggestions,
                    match_percent=st.session_state.match_percent,
                    questions=st.session_state.questions,
                    answers=st.session_state.answers,
                    feedbacks=st.session_state.feedbacks,
                    out_path="smartcv_report.pdf"
                )
                with open(path, "rb") as f:
                    st.download_button("Click to download PDF", data=f, file_name="smartcv_report.pdf", mime="application/pdf")

    # display match with color
    mp = st.session_state.match_percent or 0
    color = "green" if mp >= 75 else ("orange" if mp >= 45 else "red")
    st.markdown(f"**Match score:** <span style='color:{color}; font-weight:700'>{mp}%</span>", unsafe_allow_html=True)

    st.subheader("Suggestions to improve your CV")
    if st.session_state.suggestions:
        st.info(st.session_state.suggestions)
    else:
        st.write("No suggestions yet. Click Analyze to get suggestions.")

    st.subheader("Interview questions (from job description)")
    qs = st.session_state.questions or []
    if not qs:
        st.write("No questions generated. Click Analyze.")
    else:
        for i, q in enumerate(qs):
            st.markdown(f"**Q{i+1}. {q}**")
            # Use session_state to persist each answer
            key_ans = f"answer_{i}"
            if key_ans not in st.session_state:
                st.session_state[key_ans] = st.session_state.answers.get(i, "")
            txt = st.text_area(f"Your answer for Q{i+1}:", value=st.session_state[key_ans], key=f"ta_{i}", height=120)
            st.session_state.answers[i] = txt  # keep central mapping

            # feedback display
            fb_key = f"fb_{i}"
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("Get feedback", key=f"btn_fb_{i}"):
                    # compute and store feedback for this Q index
                    fb = openai_feedback_on_answer(q, st.session_state.answers.get(i,""), job_text, cv_text)
                    st.session_state.feedbacks[i] = fb
            with col2:
                if st.session_state.feedbacks.get(i):
                    st.success(st.session_state.feedbacks.get(i))

    st.markdown("---")
    st.subheader("Session history (this browser session)")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(f"- {h['ts']} — match: **{h['match']}%** — {len(h['questions'])} questions")
    else:
        st.write("No history yet. After analyzing, recent results will appear here.")

st.markdown("---")
st.caption("Note: For full AI features set OPENAI_API_KEY in environment. Do not paste the key into the code.")



# ----------------- Old code (commented out) -----------------
# The following code is the original app.py content before the recent edits.




# # app.py
# import os
# import streamlit as st
# import PyPDF2
# import docx2txt
# import re
# import openai

# # ---------- Helpers to extract text ----------
# def extract_text_from_pdf(uploaded_file):
#     try:
#         reader = PyPDF2.PdfReader(uploaded_file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         return text
#     except Exception as e:
#         return ""

# def extract_text_from_docx(uploaded_file):
#     try:
#         # docx2txt accepts a path or file-like; we save small temp
#         with open("temp_doc.docx", "wb") as f:
#             f.write(uploaded_file.read())
#         text = docx2txt.process("temp_doc.docx") or ""
#         os.remove("temp_doc.docx")
#         return text
#     except Exception as e:
#         return ""

# def extract_text(uploaded_file):
#     if not uploaded_file:
#         return ""
#     name = uploaded_file.name.lower()
#     if name.endswith(".pdf"):
#         return extract_text_from_pdf(uploaded_file)
#     if name.endswith(".docx"):
#         # file pointer was consumed maybe; ensure start
#         try:
#             uploaded_file.seek(0)
#         except:
#             pass
#         return extract_text_from_docx(uploaded_file)
#     # fallback: read as text
#     try:
#         uploaded_file.seek(0)
#         return uploaded_file.read().decode("utf-8")
#     except:
#         return ""

# # ---------- Basic keyword utilities for fallback ----------
# def tokenize_keywords(text):
#     text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower())
#     words = text.split()
#     # remove short words
#     words = [w for w in words if len(w) > 3]
#     freq = {}
#     for w in words:
#         freq[w] = freq.get(w, 0) + 1
#     # return top keywords
#     items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
#     return [it[0] for it in items[:40]]

# def similarity_score(job_keywords, cv_keywords):
#     if not job_keywords:
#         return 0
#     match = sum(1 for k in job_keywords if k in cv_keywords)
#     return round(100 * match / max(1, len(job_keywords)))

# # ---------- OpenAI wrapper (if key present) ----------
# def has_openai_key():
#     return bool(os.environ.get("OPENAI_API_KEY"))

# def ai_compare_and_suggest(job_text, cv_text):
#     """
#     If OPENAI_API_KEY is set, call OpenAI. Otherwise return fallback results.
#     """
#     if not job_text.strip() and not cv_text.strip():
#         return {"suggestions": "No input provided.", "questions": [], "match_percent": 0}

#     job_keywords = tokenize_keywords(job_text)
#     cv_keywords = tokenize_keywords(cv_text)
#     score = similarity_score(job_keywords, cv_keywords)

#     # Fallback simple results (no API key)
#     if not has_openai_key():
#         # produce simple suggestions
#         missing = [k for k in job_keywords if k not in cv_keywords][:12]
#         sugg_text = "Keywords to consider adding or emphasizing in your CV:\n" + ", ".join(missing) if missing else "Your CV already contains many keywords from the job description. Consider emphasizing them in bullet points."
#         # generate simple interview questions from top job keywords
#         questions = [f"Explain your experience with '{k}'." for k in job_keywords[:6]]
#         return {"suggestions": sugg_text, "questions": questions, "match_percent": score}

#     # --- If we have an OpenAI key, call the API for richer output ---
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change if needed

#     system = "You are a helpful assistant that improves CVs to match job descriptions, writes concise suggestions and generates interview questions."
#     user_prompt = (
#         f"Job description:\n{job_text}\n\nCandidate CV:\n{cv_text}\n\n"
#         "1) Give up to 8 concrete suggestions to improve the CV so it matches the job description (like specific bullets, keywords, phrasing). "
#         "2) Give a short match percentage (0-100) with a very brief reason. "
#         "3) Generate 6 likely interview questions for this role.\n\n"
#         "Respond in JSON with keys: suggestions, match_percent, questions (questions is a list).\n"
#     )

#     try:
#         resp = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user_prompt},
#             ],
#             max_tokens=700,
#             temperature=0.2,
#         )
#         text = resp["choices"][0]["message"]["content"]
#         # Try to be tolerant: if response is JSON-like, try to parse items via regex
#         # We'll do simple extraction:
#         # Find match percent
#         m = re.search(r"(\d{1,3})\s*%?", text)
#         match_percent = int(m.group(1)) if m else score
#         # Extract suggestions block (first chunk up to "questions" or so)
#         # Fallback: return full text as suggestions
#         # Extract questions by lines that end with '?'
#         questions = re.findall(r"([A-Z][^?]{5,}\?)", text)
#         suggestions = text
#         return {"suggestions": suggestions, "questions": questions[:10], "match_percent": match_percent}
#     except Exception as e:
#         # On API error fallback
#         missing = [k for k in job_keywords if k not in cv_keywords][:12]
#         sugg_text = "Could not contact AI service — fallback suggestions:\n" + ", ".join(missing)
#         questions = [f"Explain your experience with '{k}'." for k in job_keywords[:6]]
#         return {"suggestions": sugg_text, "questions": questions, "match_percent": score}

# def ai_feedback_on_answer(question, answer, job_text, cv_text):
#     if not has_openai_key():
#         # simple heuristic feedback
#         if len(answer.split()) < 8:
#             return "Your answer is short. Try to give a specific example, duration, and result. (e.g., 'I used X for Y and achieved Z')."
#         if any(word in answer.lower() for word in tokenize_keywords(job_text)[:6]):
#             return "Good: you included relevant keywords. Add a measurable result if possible (numbers, outcomes)."
#         return "Try to include a concrete example and one measurable outcome (time saved, percent improved, users impacted)."
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
#     prompt = (
#         f"Job description:\n{job_text}\n\nCandidate CV:\n{cv_text}\n\n"
#         f"Question: {question}\nCandidate's answer: {answer}\n\n"
#         "Give a short constructive feedback in 2 sentences: what was good and 1 specific improvement. "
#     )
#     try:
#         resp = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a helpful interview coach."},
#                 {"role": "user", "content": prompt},
#             ],
#             max_tokens=150,
#             temperature=0.3,
#         )
#         return resp["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         return "Could not contact AI for feedback. Try to include more concrete specifics and numbers."

# # ---------- Streamlit UI ----------
# st.set_page_config(page_title="CV Optimization Assistant", page_icon="🛠️", layout="centered")
# st.title("🛠️ CV Optimization Assistant (Simple Demo)")
# st.markdown("Upload a **job description** and your **CV**. The app will suggest improvements, generate interview questions, and let you practice answers.")

# col1, col2 = st.columns(2)
# with col1:
#     job_file = st.file_uploader("Upload job description (PDF or .txt)", type=["pdf", "txt"])
# with col2:
#     cv_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

# job_text = extract_text(job_file) if job_file else ""
# cv_text = extract_text(cv_file) if cv_file else ""

# if not job_file and not cv_file:
#     st.info("Upload sample job description and CV to start. (You can test without an API key.)")

# if st.button("Analyze CV"):
#     with st.spinner("Analyzing..."):
#         result = ai_compare_and_suggest(job_text, cv_text)
#     st.subheader("Match score")
#     st.metric("Estimated match", f"{result['match_percent']}%")
#     st.subheader("Suggestions to improve your CV")
#     st.write(result["suggestions"])
#     st.subheader("Likely interview questions")
#     questions = result.get("questions", [])
#     if not questions:
#         st.write("No questions generated.")
#     else:
#         for i, q in enumerate(questions):
#             st.write(f"**Q{i+1}.** {q}")

#     if questions:
#         st.subheader("Practice answers and get feedback")
#         for i, q in enumerate(questions):
#             ans = st.text_area(f"Your answer to Q{i+1}:", key=f"ans{i}", height=120)
#             if st.button(f"Get feedback for Q{i+1}", key=f"fb{i}"):
#                 with st.spinner("Getting feedback..."):
#                     fb = ai_feedback_on_answer(q, ans, job_text, cv_text)
#                 st.write("**Feedback:**", fb)

# st.markdown("---")
# st.write("How to get better results: (1) provide a clear job description; (2) upload a detailed CV; (3) for best AI quality set `OPENAI_API_KEY` environment variable before running.")