"""Interview Preparation Chatbot (LangGraph).

Flow:
    research -> generate_questions -> ask_question -> evaluate -> ask_ideal
        -> [generate_ideal] -> next_question -> (ask_question | summary) -> END

Run:
    uv run python interview_graph.py --role "Deep Learning Intern" --resume resume.pdf
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv(override=True)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Override via .env. Check Groq's current model list if one gets deprecated.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-120b")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "openai/gpt-oss-20b")

NUM_QUESTIONS = 5
MAX_CONTEXT_CHARS = 6000
MAX_RESUME_CHARS = 4000


@lru_cache(maxsize=1)
def get_llms():
    """Create LLM clients lazily (no import-time side effects). Reads GROQ_API_KEY from env."""
    generator = ChatGroq(model=GENERATOR_MODEL, temperature=0.7)
    evaluator = ChatGroq(model=EVALUATOR_MODEL, temperature=0)
    return generator, evaluator


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class InterviewQuestion(BaseModel):
    question: str = Field(description="The interview question text.")
    category: Literal["technical", "project", "behavioral"]


class QuestionSet(BaseModel):
    questions: List[InterviewQuestion]


class Evaluation(BaseModel):
    """Structured evaluation of the candidate's answer."""

    strengths: List[str] = Field(description="2-3 bullet points on what the candidate did well.")
    areas_for_improvement: List[str] = Field(description="2-3 actionable bullet points on what to improve.")
    overall_score: int = Field(ge=1, le=10, description="Integer score, 1 (poor) to 10 (excellent).")
    score_justification: str = Field(description="One-sentence justification for the score.")


class GraphState(TypedDict):
    job_role: str
    job_role_context: str          # web research output
    resume_summary: str            # compact text from the parsed resume
    questions: List[Dict[str, str]]    # [{question, category}]
    answers: List[str]
    evaluations: List[Dict]
    ideal_answers: List[str]
    current_q_index: int
    wants_ideal: bool
    summary: Dict


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert interviewer hiring for the role '{job_role}'. "
     "Ask realistic, specific questions - no generic trivia."),
    ("human",
     "Role research:\n{context}\n\n"
     "Candidate resume:\n{resume}\n\n"
     "Generate exactly {n} unique interview questions. Suggested mix: 2 technical questions "
     "for this role, 2 project deep-dives that reference specific projects from the resume "
     "(only if a resume is provided, otherwise make them technical), and 1 behavioral question. "
     "Do NOT include answers."),
])

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict but fair technical interviewer and coach for the role '{job_role}'. "
     "Judge clarity, technical accuracy, relevance and depth. Vague, very short or buzzword-only "
     "answers must score low."),
    ("human",
     "Role context:\n{context}\n\n"
     "Question:\n{question}\n\n"
     "Candidate's answer:\n{answer}"),
])

IDEAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an interview answer coach for the role '{job_role}'. Write a concise, well-structured "
     "ideal answer (about 120-180 words; use STAR for behavioral/project questions). "
     "Only reference experience that actually appears in the resume - never invent projects or metrics. "
     "Return only the answer text."),
    ("human",
     "Question:\n{question}\n\n"
     "Candidate's answer:\n{answer}\n\n"
     "Feedback:\n{feedback}\n\n"
     "Role context:\n{context}\n\n"
     "Resume:\n{resume}"),
])


# --------------------------------------------------------------------------- #
# Nodes (each returns a partial state update instead of mutating state)
# --------------------------------------------------------------------------- #
def research_node(state: GraphState) -> dict:
    """Gather role context from Tavily (queries run in parallel)."""
    print("\n--- 🔍 GATHERING JOB ROLE CONTEXT ---")
    job_role = state["job_role"].strip()
    if not job_role:
        raise ValueError("Job role cannot be empty.")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")
    client = TavilyClient(api_key)

    year = datetime.now().year
    queries = [
        f"key responsibilities and daily tasks of a {job_role}",
        f"essential technical skills and tools for a {job_role} in {year}",
        f"common educational background and certifications for a {job_role}",
        f"latest industry trends and future outlook for the {job_role} career path",
    ]

    def search(query: str) -> str:
        try:
            res = client.search(
                query=query,
                include_answer="advanced",
                search_depth="advanced",
                country="india",
            )
            return res.get("answer") or ""
        except Exception as e:  # one failed query shouldn't kill the rest
            print(f"  [!] Search failed for '{query}': {e}")
            return ""

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        answers = list(pool.map(search, queries))

    context = "\n\n---\n\n".join(a for a in answers if a)[:MAX_CONTEXT_CHARS]
    if context:
        print(f"--- ✅ Context gathered for '{job_role}' ---")
    else:
        print("--- ⚠️ No research context gathered; continuing with role + resume only ---")
    return {"job_role_context": context}


def generate_questions_node(state: GraphState) -> dict:
    """Generate personalized questions via structured output (no manual JSON parsing)."""
    print("\n--- 🤖 GENERATING INTERVIEW QUESTIONS ---")
    generator, _ = get_llms()
    chain = (QUESTION_PROMPT | generator.with_structured_output(QuestionSet)).with_retry(stop_after_attempt=3)

    result: QuestionSet = chain.invoke({
        "job_role": state["job_role"],
        "context": state["job_role_context"] or "(none)",
        "resume": state["resume_summary"] or "(not provided)",
        "n": NUM_QUESTIONS,
    })
    questions = [q.model_dump() for q in result.questions][:NUM_QUESTIONS]
    if not questions:
        raise RuntimeError("Model returned no questions.")

    print(f"--- ✅ {len(questions)} questions ready ---")
    return {
        "questions": questions,
        "answers": [],
        "evaluations": [],
        "ideal_answers": [""] * len(questions),
        "current_q_index": 0,
        "wants_ideal": False,
    }


def _read_answer() -> str:
    print("Type your answer. Finish with 'done' on its own line, or type 'skip' to skip.")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        cmd = line.strip().lower()
        if cmd == "done":
            break
        if cmd == "skip" and not lines:
            return ""
        lines.append(line)
    return "\n".join(lines).strip()


def ask_question_node(state: GraphState) -> dict:
    idx = state["current_q_index"]
    q = state["questions"][idx]
    print(f"\nQ{idx + 1}/{len(state['questions'])} [{q['category']}]: {q['question']}")
    answer = _read_answer()
    return {"answers": state["answers"][:idx] + [answer]}


def _print_feedback(fb: Dict) -> None:
    print("\n--- 📝 FEEDBACK ---")
    if fb.get("skipped"):
        print("Question skipped.")
        return
    if fb.get("error"):
        print(f"Evaluation failed: {fb['error']}")
        return
    print(f"Score: {fb['overall_score']}/10 - {fb['score_justification']}")
    print("\nStrengths:")
    for s in fb["strengths"]:
        print(f"  + {s}")
    print("\nImprove:")
    for s in fb["areas_for_improvement"]:
        print(f"  - {s}")


def evaluate_node(state: GraphState) -> dict:
    print("\n--- 📊 EVALUATING ANSWER ---")
    idx = state["current_q_index"]
    question = state["questions"][idx]["question"]
    answer = state["answers"][idx]

    if not answer:
        fb: Dict = {"skipped": True, "overall_score": 0}
    else:
        _, evaluator = get_llms()
        chain = (EVAL_PROMPT | evaluator.with_structured_output(Evaluation)).with_retry(stop_after_attempt=3)
        try:
            fb = chain.invoke({
                "job_role": state["job_role"],
                "context": state["job_role_context"] or "(none)",
                "question": question,
                "answer": answer,
            }).model_dump()
        except Exception as e:
            fb = {"error": str(e), "overall_score": 0}

    _print_feedback(fb)
    return {"evaluations": state["evaluations"][:idx] + [fb]}


def ask_ideal_node(state: GraphState) -> dict:
    try:
        choice = input("\nSee an ideal answer? [y/N]: ").strip().lower()
    except EOFError:
        choice = "n"
    return {"wants_ideal": choice in ("y", "yes")}


def generate_ideal_node(state: GraphState) -> dict:
    print("\n--- ✨ GENERATING IDEAL ANSWER ---")
    idx = state["current_q_index"]
    fb = state["evaluations"][idx]
    feedback_text = (
        f"Score {fb['overall_score']}/10. {fb.get('score_justification', '')}"
        if fb.get("overall_score") else "No evaluation available."
    )

    generator, _ = get_llms()
    chain = (IDEAL_PROMPT | generator).with_retry(stop_after_attempt=3)
    try:
        ideal = chain.invoke({
            "job_role": state["job_role"],
            "question": state["questions"][idx]["question"],
            "answer": state["answers"][idx] or "(skipped)",
            "feedback": feedback_text,
            "context": state["job_role_context"] or "(none)",
            "resume": state["resume_summary"] or "(not provided)",
        }).content.strip()
    except Exception as e:
        ideal = f"[Generation failed: {e}]"

    print(f"\n--- 🔎 IDEAL ANSWER for Q{idx + 1} ---\n{ideal}\n")
    ideal_answers = list(state["ideal_answers"])
    ideal_answers[idx] = ideal
    return {"ideal_answers": ideal_answers}


def next_question_node(state: GraphState) -> dict:
    return {"current_q_index": state["current_q_index"] + 1, "wants_ideal": False}


def summary_node(state: GraphState) -> dict:
    print("\n" + "=" * 60 + "\n🏁 INTERVIEW COMPLETE\n" + "=" * 60)
    scored = [(i, e["overall_score"]) for i, e in enumerate(state["evaluations"]) if e.get("overall_score")]
    for i, q in enumerate(state["questions"]):
        fb = state["evaluations"][i] if i < len(state["evaluations"]) else {}
        label = "skipped" if fb.get("skipped") else (f"{fb['overall_score']}/10" if fb.get("overall_score") else "n/a")
        print(f"Q{i + 1} [{q['category']}]: {label}")

    summary: Dict = {"answered": len(scored), "total": len(state["questions"])}
    if scored:
        avg = sum(s for _, s in scored) / len(scored)
        weakest = min(scored, key=lambda x: x[1])[0]
        summary.update(average_score=round(avg, 1), weakest_question=weakest + 1)
        print(f"\nAverage score: {avg:.1f}/10")
        print(f"Weakest answer: Q{weakest + 1} - {state['questions'][weakest]['question']}")
    return {"summary": summary}


# --------------------------------------------------------------------------- #
# Routing + graph
# --------------------------------------------------------------------------- #
def route_after_ideal_prompt(state: GraphState) -> str:
    return "generate_ideal" if state["wants_ideal"] else "next_question"


def route_after_next(state: GraphState) -> str:
    return "summary" if state["current_q_index"] >= len(state["questions"]) else "ask_question"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("research", research_node)
    g.add_node("generate_questions", generate_questions_node)
    g.add_node("ask_question", ask_question_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("ask_ideal", ask_ideal_node)
    g.add_node("generate_ideal", generate_ideal_node)
    g.add_node("next_question", next_question_node)
    g.add_node("summary", summary_node)

    g.add_edge(START, "research")
    g.add_edge("research", "generate_questions")
    g.add_edge("generate_questions", "ask_question")
    g.add_edge("ask_question", "evaluate")
    g.add_edge("evaluate", "ask_ideal")
    g.add_conditional_edges("ask_ideal", route_after_ideal_prompt,
                            {"generate_ideal": "generate_ideal", "next_question": "next_question"})
    g.add_edge("generate_ideal", "next_question")
    g.add_conditional_edges("next_question", route_after_next,
                            {"ask_question": "ask_question", "summary": "summary"})
    g.add_edge("summary", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_resume_summary(pdf_path: str) -> str:
    """Parse the resume PDF into a compact string. Returns '' on failure."""
    try:
        from extract_data_from_resume import clean_and_structure, extract_sections, extract_text_from_pdf

        data = clean_and_structure(extract_sections(extract_text_from_pdf(pdf_path)))
        return json.dumps(data, ensure_ascii=False)[:MAX_RESUME_CHARS]
    except Exception as e:
        print(f"[!] Could not parse resume ({e}); continuing without it.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Interview preparation chatbot")
    parser.add_argument("--role", default="Deep Learning Intern", help="Target job role")
    parser.add_argument("--resume", help="Path to resume PDF (optional)")
    parser.add_argument("--save", help="Save the session to this JSON file (optional)")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set.")

    init_state: GraphState = {
        "job_role": args.role,
        "job_role_context": "",
        "resume_summary": load_resume_summary(args.resume) if args.resume else "",
        "questions": [],
        "answers": [],
        "evaluations": [],
        "ideal_answers": [],
        "current_q_index": 0,
        "wants_ideal": False,
        "summary": {},
    }

    app = build_graph()
    try:
        # ~6 node visits per question; the default limit of 25 is too low for 5 questions.
        final = app.invoke(init_state, config={"recursion_limit": 10 + 6 * NUM_QUESTIONS})
    except KeyboardInterrupt:
        raise SystemExit("\nInterview cancelled.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"\nSession saved to {args.save}")


if __name__ == "__main__":
    main()
