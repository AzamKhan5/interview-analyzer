
import os
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(override=True)

API_KEY = os.getenv("Google_API_KEY", "").strip()

if not API_KEY:
    st.error(
        "Gemini API key not found. Please add Google_API_KEY "
        "to your .env file."
    )
    st.stop()

try:
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta"
        )
    )
except Exception as e:
    st.error(f"Unable to initialize Gemini client: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE = 5 * 1024 * 1024


# =====================================================
# 2. STRUCTURED OUTPUT SCHEMA
# =====================================================

class CandidateProfile(BaseModel):
    inferred_role: str = Field(
        description="Most likely professional role based on the interview."
    )
    seniority_level: str = Field(
        description="Estimated seniority level based on evidence."
    )
    rationale: str = Field(
        description="Reasoning supporting the inferred role and seniority."
    )
    confidence: str = Field(
        description="High, Medium, or Low confidence in the profile inference."
    )


class EvidenceItem(BaseModel):
    observation: str = Field(
        description="A specific professional strength or potential concern."
    )
    evidence: str = Field(
        description="Supporting evidence from the interview transcript."
    )


class InterviewAnalysis(BaseModel):
    topics_covered: list[str] = Field(
        description="Main subjects discussed during the interview."
    )

    candidate_profile: CandidateProfile

    strengths: list[EvidenceItem] = Field(
        description="Professional strengths supported by evidence."
    )

    concerns_or_gaps: list[EvidenceItem] = Field(
        description="Potential gaps or areas requiring further validation."
    )

    candidate_summary: str = Field(
        description="Professional candidate summary in 3 to 6 sentences."
    )

    follow_up_questions: list[str] = Field(
        description="Suggested questions for the next interview round."
    )


# =====================================================
# 3. PROFESSIONAL HR PROMPT
# =====================================================

def build_prompt(transcript, job_description):

    job_context = (
        job_description.strip()
        if job_description.strip()
        else "No job description provided. Analyze using only "
             "the interview transcript."
    )

    prompt = f"""
You are a senior HR analyst and professional interview
assessment specialist.

Analyze the interview transcript and produce a structured,
objective, evidence-based candidate assessment.

IMPORTANT GUIDELINES:

1. OBJECTIVITY
- Use only information explicitly available in the transcript.
- Never invent qualifications, experience, achievements, or skills.
- Distinguish demonstrated abilities from unverified claims.
- Clearly identify missing information.

2. FAIRNESS
- Evaluate only job-related professional information.
- Do not infer or evaluate age, gender, race, religion,
  disability, marital status, nationality, or other protected
  characteristics.
- Do not make a final hiring or rejection decision.
- Avoid subjective personality judgments.

3. EVIDENCE-BASED ASSESSMENT
- Support strengths with relevant interview examples.
- Identify unclear responses and potential skill gaps.
- An unanswered question is not proof of incompetence.
- Distinguish facts from areas requiring verification.
- Never fabricate direct quotations.

4. PROFESSIONAL COMMUNICATION
- Use clear and concise HR language.
- Avoid repetition and exaggerated statements.
- Maintain a neutral and professional tone.

ANALYSIS REQUIREMENTS:

A. TOPICS COVERED
Identify the major technical, professional, and
experience-related topics discussed.

B. CANDIDATE PROFILE
Identify the most likely professional role and seniority.
Explain the reasoning using transcript evidence.
Communicate uncertainty when evidence is insufficient.

C. STRENGTHS
Identify demonstrated professional strengths.
For each strength, provide a supporting observation
and evidence from the transcript.

D. CONCERNS AND GAPS
Identify unclear responses, missing information,
potential skill gaps, and competencies requiring validation.
Explain the evidence or information gap for each concern.
Do not present speculation as fact.

E. CANDIDATE SUMMARY
Write a professional summary of 3 to 6 sentences.
Include:
- Professional background
- Demonstrated strengths
- Areas requiring further assessment
- Overall interview observations

F. FOLLOW-UP QUESTIONS
Suggest relevant questions for the next interview.
Focus on validating important skills and resolving
identified information gaps.

JOB DESCRIPTION / ROLE REQUIREMENTS:

{job_context}

INTERVIEW TRANSCRIPT:

{transcript}

Return the result strictly according to the supplied
structured output schema.
"""

    return prompt


# =====================================================
# 4. GEMINI ANALYSIS
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
# 5. MARKDOWN REPORT GENERATION
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

    if result.strengths:
        for item in result.strengths:
            report.append(f"### {item.observation}")
            report.append(f"Evidence: {item.evidence}")
    else:
        report.append("No specific strengths were established.")

    report.append("\n## 4. Concerns and Areas to Validate\n")

    if result.concerns_or_gaps:
        for item in result.concerns_or_gaps:
            report.append(f"### {item.observation}")
            report.append(f"Evidence: {item.evidence}")
    else:
        report.append("No specific concerns were established.")

    report.append("\n## 5. Candidate Summary\n")
    report.append(result.candidate_summary)

    report.append("\n## 6. Suggested Follow-up Questions\n")

    for question in result.follow_up_questions:
        report.append(f"- {question}")

    report.append(
        "\n---\n*AI-generated decision-support report. "
        "Verify all observations against the original transcript. "
        "This report does not make a hiring recommendation.*"
    )

    return "\n".join(report)


# =====================================================
# 6. APPLICATION HEADER
# =====================================================

st.title("🎯 AI Interview Analyzer")

st.markdown(
    """
    Analyze interview transcripts using Google Gemini AI
    and generate structured, evidence-based candidate reports.

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
        AI-powered interview transcript analysis
        for professional recruitment workflows.
        """
    )

    st.divider()

    st.subheader("Analysis Sections")

    st.markdown(
        """
        - Topics Covered
        - Candidate Profile
        - Candidate Strengths
        - Concerns and Skill Gaps
        - Candidate Summary
        - Follow-up Questions
        """
    )

    st.divider()

    st.caption(f"AI Model: {MODEL_NAME}")
    st.caption("Maximum upload size: 5 MB")


# =====================================================
# 8. TRANSCRIPT UPLOAD
# =====================================================

st.subheader("📄 Upload Interview Transcript")

uploaded_file = st.file_uploader(
    "Choose a transcript file",
    type=["txt"],
    help="Upload a UTF-8 encoded interview transcript."
)

job_description = st.text_area(
    "Job Description / Role Requirements (Optional)",
    placeholder=(
        "Paste the job description or key role requirements. "
        "This helps contextualize the candidate's experience."
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
            "Unable to read the file. Please upload a "
            "UTF-8 encoded text file."
        )
        st.stop()

    if not transcript:
        st.warning("The uploaded transcript is empty.")
        st.stop()

    word_count = len(transcript.split())

    col1, col2, col3 = st.columns(3)

    col1.metric("Word Count", f"{word_count:,}")
    col2.metric("Characters", f"{len(transcript):,}")
    col3.metric(
        "File Size",
        f"{len(file_content) / 1024:.1f} KB"
    )

    with st.expander("Preview Interview Transcript"):

        st.text_area(
            "Transcript Preview",
            value=transcript[:5000],
            height=250,
            disabled=True
        )

        if len(transcript) > 5000:
            st.caption("Showing the first 5,000 characters.")

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
                "The transcript is too short for meaningful analysis."
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

                st.success(
                    "Interview analysis generated successfully!"
                )

            except Exception as e:

                error_message = str(e)
                error_lower = error_message.lower()

                if (
                    "api_key_invalid" in error_lower
                    or "api key not valid" in error_lower
                    or "api_key" in error_lower
                ):
                    st.error(
                        "Invalid Gemini API key. Check your .env file "
                        "and generate a valid key from Google AI Studio."
                    )

                elif "permission_denied" in error_lower:
                    st.error(
                        "Gemini API access denied. Check your API key "
                        "permissions and API configuration."
                    )

                elif "resource_exhausted" in error_lower or "429" in error_message:
                    st.error(
                        "Gemini API quota or rate limit exceeded. "
                        "Check your API usage and try again later."
                    )

                else:
                    st.error(
                        f"Unable to generate analysis: {error_message}"
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

    st.caption(f"Profile Confidence: {profile.confidence}")

    st.write(profile.rationale)

    # Topics Covered

    st.subheader("2. Topics Covered")

    for topic in result.topics_covered:
        st.markdown(f"- {topic}")

    # Strengths and Concerns

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

        st.subheader("4. Concerns and Skill Gaps")

        if result.concerns_or_gaps:

            for item in result.concerns_or_gaps:

                with st.container(border=True):

                    st.markdown(f"**{item.observation}**")
                    st.write(item.evidence)

        else:
            st.info("No specific concerns were established.")

    # Candidate Summary

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

    # =================================================
    # 11. DOWNLOAD REPORT
    # =================================================

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
        "Important: This is an AI-generated decision-support report. "
        "HR professionals must independently verify the evidence "
        "and make all recruitment decisions."
    )
