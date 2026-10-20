"""Resume PDF parser: turns a resume into a structured list of sections.

Output format (unchanged):
    [{"section": "skills", "data": ["Python", ...]}, {"section": "projects", "data": [...]}, ...]

Run:
    uv run python extract_data_from_resume.py resume.pdf
"""
import argparse
import json
import re
from pprint import pprint

from pypdf import PdfReader  # PyPDF2 is deprecated; `uv add pypdf`

# Order matters: the consumer truncates long resumes, so the most useful sections come first.
SECTION_ORDER = ["summary", "skills", "projects", "experience", "education", "certifications", "achievements"]

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "technologies", "tech stack"],
    "projects": ["projects", "personal projects", "academic projects", "key projects", "selected projects"],
    "experience": ["experience", "work experience", "professional experience", "work history",
                   "internships", "internship experience", "employment"],
    "education": ["education", "academics", "academic background", "qualifications"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "courses"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "awards & achievements",
                     "positions of responsibility", "extracurricular activities"],
}
_ALIAS_TO_SECTION = {a: sec for sec, aliases in SECTION_ALIASES.items() for a in aliases}

# Bullet glyphs (incl. the private-use one many PDFs emit) or a spaced hyphen/dash.
BULLET_RE = re.compile(r"^\s*(?:[•●▪■◦○·\uf0b7*]\s*|[-–—]\s+)")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF."""
    reader = PdfReader(pdf_path)
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if not text:
        raise ValueError(f"No extractable text in '{pdf_path}' (scanned PDF? it would need OCR).")
    return text


def _match_header(line: str):
    """Return (section, inline_remainder) if the line is a section header, else None.

    Only matches when the line's heading part *equals* a known header, so a bullet like
    'Built a Projects dashboard' or 'Developed skills in...' is no longer mistaken for one.
    """
    if len(line) > 80:
        return None
    head, _, rest = line.partition(":")
    key = re.sub(r"[^a-z& ]", "", head.lower())
    key = re.sub(r"\s+", " ", key).strip()
    section = _ALIAS_TO_SECTION.get(key)
    return (section, rest.strip()) if section else None


def extract_sections(text: str) -> dict:
    """Split resume text into {section: [lines]} using exact header matches."""
    sections: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = None if BULLET_RE.match(line) else _match_header(line)
        if header:
            current, rest = header
            sections.setdefault(current, [])  # don't wipe a section that appears twice
            if rest:
                sections[current].append(rest)
        elif current:
            sections[current].append(line)
    return sections


def _merge_wrapped_lines(lines):
    """Strip bullets and re-join lines that PDF layout wrapped mid-sentence."""
    items, prev_was_bullet = [], False
    for line in lines:
        if BULLET_RE.match(line):
            items.append(BULLET_RE.sub("", line).strip())
            prev_was_bullet = True
        elif prev_was_bullet and items and not items[-1].endswith("."):
            items[-1] += " " + line  # continuation of the previous bullet
        else:
            items.append(line)  # title / standalone line
            prev_was_bullet = False
    return [i for i in items if i]


def _parse_skills(lines):
    """Flatten skill lines like 'Languages: Python, C++ | SQL' into a deduped list."""
    skills, seen = [], set()
    for line in lines:
        line = BULLET_RE.sub("", line)
        if ":" in line:  # drop the category label ("Languages:", "Frameworks:")
            line = line.split(":", 1)[1]
        for s in re.split(r"[,|;•·]", line):
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                skills.append(s)
    return skills


def clean_and_structure(sections: dict) -> list:
    """Convert extracted sections into a list of {'section', 'data'} dicts."""
    structured = []
    for name in SECTION_ORDER:
        lines = sections.get(name)
        if not lines:
            continue
        if name == "skills":
            data = _parse_skills(lines)
        elif name == "summary":
            data = [" ".join(lines)]
        else:
            data = _merge_wrapped_lines(lines)
        if data:
            structured.append({"section": name, "data": data})
    return structured


def parse_resume(pdf_path: str) -> list:
    """One-call helper: PDF path -> structured sections."""
    return clean_and_structure(extract_sections(extract_text_from_pdf(pdf_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a resume PDF into structured sections")
    parser.add_argument("pdf_path", help="Path to the resume PDF")
    parser.add_argument("--json", action="store_true", help="Print as JSON instead of pretty-print")
    args = parser.parse_args()

    data = parse_resume(args.pdf_path)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        pprint(data)
