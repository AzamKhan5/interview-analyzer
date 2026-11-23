# AI Interview Analyzer 🎯

An AI-powered interview transcript analysis application developed as part of the **Cisco Apprenticeship Program**. The application uses Google Gemini AI to analyze interview transcripts and generate structured, evidence-based candidate assessment reports, helping HR professionals streamline interview analysis.

## 🚀 Project Overview

The AI Interview Analyzer allows recruiters and HR professionals to upload interview transcripts and generate comprehensive summaries using Generative AI.

It analyzes candidate responses, identifies professional strengths and potential skill gaps, infers suitable professional profiles, and suggests relevant follow-up interview questions.

The application is built using Python and Streamlit and deployed on Streamlit Community Cloud.

## ✨ Key Features

- **Transcript Upload:** Upload interview transcripts in `.txt` format.
- **Transcript Preview:** View uploaded content along with word count, character count, and file size.
- **AI-Powered Analysis:** Uses Google Gemini 2.5 Flash to analyze interview conversations.
- **Topics Covered:** Identifies the major technical and professional topics discussed.
- **Candidate Profiling:** Infers the candidate's professional role and potential seniority level.
- **Strength Identification:** Highlights candidate strengths with supporting interview evidence.
- **Skill Gap Analysis:** Identifies potential concerns and areas requiring further validation.
- **Candidate Summary:** Generates a concise, structured candidate overview.
- **Follow-up Questions:** Recommends relevant questions for subsequent interview rounds.
- **Job Description Context:** Supports optional job descriptions for contextualized analysis.
- **Report Export:** Allows users to download reports in Markdown and JSON formats.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| Google Gemini 2.5 Flash | AI-powered transcript analysis |
| Google GenAI SDK | Gemini API integration |
| Prompt Engineering | Structured and evidence-based analysis |
| Pydantic | Structured JSON output validation |
| Python-dotenv | Secure API key configuration |

## 🧠 AI and Prompt Engineering

The application uses structured prompt engineering to guide Gemini in analyzing interview transcripts.

The prompts emphasize:
- Objective and evidence-based candidate analysis.
- Professional strengths and potential skill gaps.
- Candidate profiling and seniority estimation.
- Fair and job-related evaluation.
- Relevant follow-up interview questions.

Structured JSON responses are validated using Pydantic to ensure consistent output and improve report generation.

## 📂 Project Structure

```text
AI-Interview-Analyzer/
│
├── app.py
├── .env
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AzamKhan5/interview-analyzer.git
cd AI-Interview-Analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Create a `.env` file in the project directory:

```env
Google_API_KEY=your_gemini_api_key
```

Obtain your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment involves connecting the GitHub repository to Streamlit and configuring the Gemini API key through Streamlit Secrets.

Example Streamlit secret:

```toml
Google_API_KEY = "your_gemini_api_key"
```

For deployment, configure the application to read the API key from Streamlit Secrets, with the `.env` file retained for local development.

## 🔒 Security and Responsible AI

- API keys are stored separately from application source code.
- Interview analysis focuses on job-related professional information.
- AI-generated observations should be verified against the original transcript.
- The application provides decision-support insights and does not make automated hiring or rejection decisions.

## 🎯 Project Outcome

Successfully developed and deployed an AI-powered interview analysis application as part of the Cisco Apprenticeship Program, demonstrating practical implementation of Generative AI, prompt engineering, Python development, API integration, structured data validation, and cloud deployment.

---

**Developed as part of the Cisco Apprenticeship Program.**

*Technologies: Python | Streamlit | Google Gemini AI | Prompt Engineering | Pydantic*
