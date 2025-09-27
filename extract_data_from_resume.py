import re
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_sections(text):
    """Split resume into sections based on keywords."""
    sections = {}
    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        if re.search(r"\bSkills\b", line, re.I):
            current_section = "skills"
            sections[current_section] = []
        elif re.search(r"\bProjects?\b", line, re.I):
            current_section = "projects"
            sections[current_section] = []
        elif re.search(r"\bExperience\b|\bWork History\b", line, re.I):
            current_section = "experience"
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)

    return sections

def clean_and_structure(sections):
    """Convert extracted sections into list of dicts."""
    structured_data = []

    if "skills" in sections:
        skills = re.split(r",|\|", " ".join(sections["skills"]))
        skills = [s.strip() for s in skills if s.strip()]
        structured_data.append({"section": "skills", "data": skills})

    if "projects" in sections:
        structured_data.append({"section": "projects", "data": sections["projects"]})

    if "experience" in sections:
        structured_data.append({"section": "experience", "data": sections["experience"]})

    return structured_data

# ---- Example usage ----
if __name__ == "__main__":
    pdf_path = "C:/python_code/NLPGenAIcourse/LangGraph/hr_resume_selector/resumes/resume_20_09.pdf"  # replace with your resume file
    text = extract_text_from_pdf(pdf_path)
    sections = extract_sections(text)
    data = clean_and_structure(sections)

    from pprint import pprint
    pprint(data)

    