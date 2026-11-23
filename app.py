
import os
import json
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

API_KEY = os.getenv("Google_API_KEY")

if not API_KEY:
    st.error("Gemini API key not found. Please configure it in your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# =====================================================
# 2. STRUCTURED OUTPUT SCHEMA
# =====================================================

class CandidateProfile(BaseModel):
    inferred_role: str = Field(
        description="Most likely professional role based on the interview."
    )
    seniority_level: str = Field(
        description="Estimated professional seniority, if supported by evidence."
    )
    rationale: str = Field(
        description="Explanation supporting the inferred role and seniority."
    )
    confidence: str = Field(
        description="High, Medium, or Low confidence in this profile inference."
    )


class EvidenceItem(BaseModel):
    observation: str = Field(
        description="Specific strength or potential concern."
    )
    evidence: str = Field(
        description="Relevant statement or example from the transcript."
    )


class InterviewAnalysis(BaseModel):
    topics_covered: list[str] = Field(
        description="Main themes and subjects discussed during the interview."
    )

    candidate_profile: CandidateProfile

    strengths: list[EvidenceItem] = Field(
        description="Candidate strengths supported by interview evidence."
    )

    concerns_or_gaps: list[EvidenceItem] = Field(
        description="Potential skill gaps, unclear answers, or areas requiring validation."
    )

    candidate_summary: str = Field(
        description="A professional summary of the candidate in 3 to 6 sentences."
    )

    follow_up_questions: list[str] = Field(
        description="Suggested questions for the next interview round."
    )


# =====================================================
# 3. PROMPT ENGINEERING
# =====================================================

def build_prompt(transcript, job_description):

    job_context = job_description.strip() if job_description.strip() else (
        "No job description provided. Analyze the candidate's professional "
        "background based only on the interview transcript."
    )

    prompt = f"""
You are a senior HR analyst and professional interview assessment specialist.

Your task is to analyze the interview transcript and generate an objective,
structured, evidence-based interview report.

IMPORTANT ANALYSIS GUIDELINES:

1. OBJECTIVITY
- Base every observation on information explicitly available in the transcript.
- Do not invent qualifications, experience, achievements, or skills.
- Distinguish between demonstrated abilities and unverified claims.
- If information is missing, clearly state that it was not established.

2. FAIRNESS
- Evaluate only job-related professional information.
- Do not infer or evaluate age, gender, race, religion, disability,
  marital status, nationality, or other protected characteristics.
- Do not make a final hiring or rejection decision.
- Avoid subjective personality judgments and unsupported assumptions.

3. EVIDENCE-BASED ASSESSMENT
- Support strengths with relevant examples from the interview.
- Identify unclear responses, missing information, and potential skill gaps.
- Do not interpret an unanswered question as proof of incompetence.
- Clearly distinguish observed evidence from areas requiring verification.

4. PROFESSIONAL WRITING
- Use clear, concise, professional HR language.
- Avoid unnecessary repetition.
- Keep all observations relevant to recruitment.
- Do not use exaggerated or promotional language.

ANALYSIS REQUIREMENTS:

A. TOPICS COVERED
Identify the major technical, professional, and experience-related topics
discussed during the interview.

B. CANDIDATE PROFILE
Identify the most likely professional role and seniority level.
Explain your reasoning using transcript evidence.
If the available evidence is insufficient, communicate uncertainty.

C. STRENGTHS
Identify demonstrated professional strengths.
For each strength, provide a supporting observation or relevant evidence
from the transcript.

D. CONCERNS OR GAPS
Identify potential weaknesses, missing information, unclear responses,
or competencies that require further validation.
For each item, provide supporting evidence or explain what is missing.
Do not present speculation as fact.

E. CANDIDATE SUMMARY
Write a concise professional summary of 3 to 6 sentences.
Include:
- Professional background
- Demonstrated strengths
- Areas requiring further assessment
- Overall interview impression

F. FOLLOW-UP QUESTIONS
Suggest specific, relevant questions that HR or the hiring manager can
ask in the next interview round.
Focus on validating important skills and resolving identified gaps.

JOB DESCRIPTION / ROLE CONTEXT:

{job_context}

INTERVIEW TRANSCRIPT:

{transcript}

Return the analysis strictly according to the provided structured output schema.
"""


    return prompt


# =====================================================
# 4. GEMINI ANALYSIS FUNCTION
# =====================================================

def analyze_interview(transcript, job_description):

    prompt = build_prompt(transcript, job_description)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InterviewAnalysis,
            temperature=0.2
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return InterviewAnalysis.model_validate_json(response.text)


# =====================================================
# 5. REPORT GENERATION
# =====================================================

def generate_markdown_report(result):

    report = []

    report.append("# Interview Analysis Report\n")

    report.append("## 1. Topics Covered\n")

    for topic in result.topics_covered:
        report.append(f"- {topic}")

    profile = result.candidate_profile

    report.append("\n## 2. Candidate Profile\n")
    report.append(f"**Inferred Role:** {profile.inferred_role}")
    report.append(f"**Seniority Level:** {profile.seniority_level}")
    report.append(f"**Confidence:** {profile.confidence}")
    report.append(f"**Rationale:** {profile.rationale}")

    report.append("\n## 3. Candidate Strengths\n")

    for item in result.strengths:
        report.append(f"### {item.observation}")
        report.append(f"Evidence: {item.evidence}")

    report.append("\n## 4. Concerns and Areas to Validate\n")

    for item in result.concerns_or_gaps:
        report.append(f"### {item.observation}")
        report.append(f"Evidence: {item.evidence}")

    report.append("\n## 5. Candidate Summary\n")
    report.append(result.candidate_summary)

    report.append("\n## 6. Suggested Follow-up Questions\n")

    for question in result.follow_up_questions:
        report.append(f"- {question}")

    report.append(
        "\n---\n*AI-generated analysis. Verify all observations "
        "against the original interview transcript before making "
        "recruitment decisions.*"
    )

    return "\n".join(report)


# =====================================================
# 6. APPLICATION HEADER
# =====================================================

st.title("🎯 AI Interview Analyzer")

st.markdown(
    """
    Analyze interview transcripts using Gemini AI and generate
    structured, evidence-based candidate reports.

    **Designed for HR professionals, recruiters, and hiring managers.**
    """
)

st.divider()


# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:

    st.header("About the Application")

    st.write(
        """
        This application uses Google Gemini to analyze interview
        transcripts and organize candidate information into
        structured HR reports.
        """
    )

    st.divider()

    st.subheader("Analysis Sections")

    st.markdown(
        """
        - Topics Covered
        - Candidate Profile
        - Strengths
        - Concerns and Gaps
        - Candidate Summary
        - Follow-up Questions
        """
    )

    st.divider()

    st.caption("AI Model: Gemini 2.5 Flash")
    st.caption("Maximum upload size: 5 MB")


# =====================================================
# 8. TRANSCRIPT UPLOAD
# =====================================================

st.subheader("📄 Upload Interview Transcript")

uploaded_file = st.file_uploader(
    "Choose a transcript file",
    type=["txt"],
    help="Upload a plain text interview transcript (UTF-8)."
)

job_description = st.text_area(
    "Job Description / Role Requirements (Optional)",
    placeholder=(
        "Paste the job description or key role requirements here. "
        "This helps the AI contextualize the candidate's experience."
    ),
    height=150
)


if uploaded_file is not None:

    file_content = uploaded_file.getvalue()

    if len(file_content) > MAX_FILE_SIZE:

        st.error("File exceeds the maximum allowed size of 5 MB.")
        st.stop()

    try:
        transcript = file_content.decode("utf-8-sig").strip()

    except UnicodeDecodeError:

        st.error(
            "Unable to read the file. Please upload a UTF-8 encoded text file."
        )
        st.stop()

    if not transcript:

        st.warning("The uploaded transcript is empty.")
        st.stop()

    word_count = len(transcript.split())

    col1, col2, col3 = st.columns(3)

    col1.metric("Word Count", f"{word_count:,}")
    col2.metric("Characters", f"{len(transcript):,}")
    col3.metric("File Size", f"{len(file_content) / 1024:.1f} KB")

    with st.expander("Preview Interview Transcript"):

        st.text_area(
            "Transcript Preview",
            value=transcript[:5000],
            height=250,
            disabled=True
        )

        if len(transcript) > 5000:
            st.caption("Preview limited to the first 5,000 characters.")

    st.divider()


    # =================================================
    # 9. GENERATE ANALYSIS
    # =================================================

    if st.button(
        "🚀 Generate Interview Analysis",
        type="primary",
        use_container_width=True
    ):

        if word_count < 20:

            st.warning(
                "The transcript appears too short for a meaningful analysis."
            )

        else:

            try:

                with st.spinner(
                    "Gemini is analyzing the interview transcript..."
                ):

                    result = analyze_interview(
                        transcript,
                        job_description
                    )

                    st.session_state["interview_analysis"] = (
                        result.model_dump()
                    )

                st.success("Interview analysis generated successfully!")

            except Exception as e:

                st.error(
                    f"Unable to generate analysis. Please try again. "
                    f"Error: {str(e)}"
                )


# =====================================================
# 10. DISPLAY GENERATED REPORT
# =====================================================

if "interview_analysis" in st.session_state:

    result = InterviewAnalysis.model_validate(
        st.session_state["interview_analysis"]
    )

    st.divider()

    st.header("📊 Interview Analysis Report")

    # Candidate Profile

    st.subheader("1. Candidate Profile")

    profile = result.candidate_profile

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Inferred Role", profile.inferred_role)

    with col2:
        st.metric("Seniority Level", profile.seniority_level)

    st.caption(f"Profile confidence: {profile.confidence}")

    st.write(profile.rationale)

    # Topics

    st.subheader("2. Topics Covered")

    for topic in result.topics_covered:
        st.markdown(f"- {topic}")

    # Strengths and concerns

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("3. Candidate Strengths")

        if result.strengths:

            for item in result.strengths:

                with st.container(border=True):

                    st.markdown(f"**{item.observation}**")
                    st.write(item.evidence)

        else:
            st.info("No specific strengths were established.")

    with col2:

        st.subheader("4. Concerns and Gaps")

        if result.concerns_or_gaps:

            for item in result.concerns_or_gaps:

                with st.container(border=True):

                    st.markdown(f"**{item.observation}**")
                    st.write(item.evidence)

        else:
            st.info("No specific concerns were established.")

    # Summary

    st.subheader("5. Candidate Summary")

    with st.container(border=True):

        st.write(result.candidate_summary)

    # Follow-up Questions

    st.subheader("6. Suggested Follow-up Questions")

    for index, question in enumerate(
        result.follow_up_questions,
        start=1
    ):

        st.markdown(f"**{index}.** {question}")

    # Download

    st.divider()

    st.subheader("📥 Export Interview Report")

    report = generate_markdown_report(result)

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="Download HR Report (.md)",
            data=report,
            file_name="interview_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="Download Analysis (.json)",
            data=result.model_dump_json(indent=4),
            file_name="interview_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    st.caption(
        "Important: This report is AI-generated decision support, "
        "not a hiring recommendation. HR professionals should "
        "independently verify the evidence and make all recruitment decisions."
    )
