from langgraph.graph import StateGraph, START, END
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ChatMessage
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

#groq
groq_api_key = os.getenv("GROQ_API_KEY")
generator_llm = ChatGroq(model ="openai/gpt-oss-120b",groq_api_key=groq_api_key)
evaluator_llm = ChatGroq(model ="Gemma2-9b-It",groq_api_key=groq_api_key)

from pydantic import BaseModel, Field, conint
from typing import List


#--------------------------------------------------------------------------------------------------------------
class Evaluation(BaseModel):
    """A structured evaluation of the user's interview answer."""
    strengths: List[str] = Field(
        description="A list of 2-3 bullet points highlighting what the user did well."
    )
    areas_for_improvement: List[str] = Field(
        description="A list of 2-3 actionable bullet points for what could be improved."
    )
    overall_score: conint(ge=1, le=10) = Field(
        description="A single integer score from 1 (poor) to 10 (excellent)."
    )
    score_justification: str = Field(
        description="A concise, one-sentence justification for the given score."
    )


class GraphState(TypedDict):
    job_role: str # User input
    job_role_context: str # from research node
    candidate_data: List[Dict[str, List[str]]] #parsed from candidate resume
    ideal_answers: List[str]  # generated ideal answers
    generated_qas: List[Dict[str, str]]   # [{question, answer}]
    evaluation_feedback: List[Dict] # from evaluator node
    current_q_index: int # which question we're on
    wants_ideal: bool # whether the user wants ideal answers
    num_steps: int # how many steps taken
    max_iterations: int # max allowed steps


#--------------------------------------------------------------------------------------------------------------

def generator_node(state: GraphState) -> GraphState:
    """
    Uses the LLM (generator_llm) to create interview questions
    (without answers) based on the job role context gathered earlier.
    """
    print("\n--- 🤖 GENERATING INTERVIEW QUESTIONS ONLY ---")

    # Ensure we have context
    job_context = state.get("job_role_context", "")
    job_role = state.get("job_role", "")

    if not job_context:
        raise ValueError("No job_role_context found. Research node must run first.")

    # Prompt for the LLM
    query = f"""
    You are an expert interviewer.
    Generate exactly 5 unique interview questions for the role '{job_role}'.
    Do NOT provide answers. 
    Respond in strict JSON format as a list of objects, 
    each with keys: "question" and "answer". 
    Leave "answer" as an empty string.
    Example:
    [
      {{"question": "What is X?", "answer": ""}},
      {{"question": "How would you handle Y?", "answer": ""}}
    ]

    Context:\n\n{job_context}
    """

    # Call the LLM
    try:
        response = generator_llm.invoke([HumanMessage(content=query)])
        output_text = response.content
    except Exception as e:
        print(f"[!] Error calling generator_llm: {e}")
        output_text = "[]"

    # Try parsing into JSON
    import json
    try:
        qas = json.loads(output_text)
    except Exception:
        print("[!] Model did not return valid JSON, falling back...")
        qas = [{"question": output_text, "answer": ""}]

    # Save into state
    state["generated_qas"] = qas
    state["num_steps"] = state.get("num_steps", 0) + 1

    print("--- ✅ QUESTION GENERATION COMPLETE ---")
    return state

def research_node(state: GraphState) -> GraphState:

    """
    Performs targeted web research on a given job role to gather context using the Tavily client.
    
    This node's single responsibility is to collect general information about the job role,
    which will be used by downstream nodes.
    """
    
    print("--- 🔍 GATHERING JOB ROLE CONTEXT (Tavily Search) ---")

    # 1. Extract the job role from the state
    job_role = state['job_role']
    if not job_role:
        raise ValueError("Job role cannot be empty.")

    # 2. Initialize the Tavily client
    # Note: Ensure the TAVILY_API_KEY environment variable is set.
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")

        tavily_client = TavilyClient(tavily_api_key)
    except Exception as e:
        print(f"Error initializing Tavily client: {e}")
        raise

    # 3. Define a set of focused queries to build a comprehensive profile of the job role
    search_queries = [
        f"key responsibilities and daily tasks of a {job_role}",
        f"essential technical skills and required software tools for a {job_role} in 2025",
        f"common educational background and certifications for a {job_role}",
        f"latest industry trends and future outlook for the {job_role} career path"
    ]

    print(f"Performing {len(search_queries)} targeted searches for the role: '{job_role}'")
    
    # 4. Execute searches and aggregate the content from the results
    all_context = []
    try:
        for query in search_queries:
            response = tavily_client.search(
            query=query,
            include_answer="advanced",
            search_depth="advanced",
            country="india"
        )
            all_context.append(response.get('answer', []))

    except Exception as e:
        print(f"    [!] Search failed for query '{query}': {e}")

    # 5. Join the collected text into a single, comprehensive context string
    job_role_context = "\n\n---\n\n".join(all_context)
    
    if not job_role_context:
        print("--- ⚠️ RESEARCH WARNING: No context was gathered. Downstream tasks may fail. ---")
    else:
        print(f"--- ✅ RESEARCH COMPLETE: Successfully gathered context for '{job_role}'. ---")
    
    # 6. Update the state with the new context and increment the step counter
    state['job_role_context'] = job_role_context
    state['num_steps'] = state.get('num_steps', 0) + 1
    
    return state


def interview_chat(state: GraphState) -> GraphState:
    """
    Handles the interview interaction - asks question and collects user answer.
    """
    qas = state.get("generated_qas", [])
    idx = state.get("current_q_index", 0)

    if idx >= len(qas):
        print("✅ Interview complete! All questions answered.")
        return state

    question = qas[idx].get("question", "")
    print(f"\nQ{idx+1}: {question}")

    # Collect user answer in multi-line style. User types 'done' when finished.
    lines = []
    print("Type your answer. Type a blank line then confirm, or type 'done' on a new line to finish.")
    while True:
        chunk = input()
        if chunk.strip().lower() == "done":
            break
        # blank line = ask if done
        if chunk.strip() == "":
            yn = input("It looks like you pressed Enter. Are you done answering? (yes/no): ").strip().lower()
            if yn in ("yes", "y"):
                break
            else:
                continue
        lines.append(chunk)

    user_answer = " ".join(lines).strip()
    qas[idx]["answer"] = user_answer
    state["generated_qas"] = qas

    return state


def evaluator_node(state: GraphState) -> GraphState:
    """
    Evaluates the user's answer using an LLM and provides structured feedback.
    """
    print("\n--- 📊 EVALUATING ANSWER ---")

    idx = state.get("current_q_index", 0)
    qas = state.get("generated_qas", [])
    
    if idx >= len(qas):
        print("[!] evaluator_node: current_q_index out of range.")
        return state

    # Get required data from state
    question = qas[idx].get("question", "")
    user_answer = qas[idx].get("answer", "")
    job_role_context = state.get("job_role_context", "")
    job_role = state.get("job_role", "")

    # Setup the output parser
    parser = JsonOutputParser(pydantic_object=Evaluation)

    # Create the prompt for the evaluator LLM
    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert technical interviewer and performance coach for the role of '{job_role}'.
        Your task is to provide a fair, structured evaluation of the candidate's answer.

        **Job Role Context:**
        {job_context}

        **Interview Question:**
        {question}

        **Candidate's Answer:**
        {user_answer}
        
        **Instructions:**
        Evaluate the answer based on clarity, technical accuracy, relevance to the role, and depth.
        Provide your feedback in the required JSON format.

        {format_instructions}
        """
    )
    
    # Create and invoke the evaluation chain
    evaluator_chain = prompt_template | evaluator_llm | parser

    try:
        feedback = evaluator_chain.invoke({
            "job_role": job_role,
            "job_context": job_role_context,
            "question": question,
            "user_answer": user_answer,
            "format_instructions": parser.get_format_instructions(),
        })

        # Ensure evaluation_feedback is a list
        if "evaluation_feedback" not in state or not isinstance(state["evaluation_feedback"], list):
            state["evaluation_feedback"] = []

        # Expand list size if needed
        while len(state["evaluation_feedback"]) <= idx:
            state["evaluation_feedback"].append({})

        # Store at index
        state["evaluation_feedback"][idx] = feedback

        print("\n--- 📝 FEEDBACK ---")
        print(f"Overall Score: {feedback['overall_score']}/10")
        print(f"Justification: {feedback['score_justification']}")
        print("\nStrengths:")
        for strength in feedback['strengths']:
            print(f"  - {strength}")
        print("\nAreas for Improvement:")
        for improvement in feedback['areas_for_improvement']:
            print(f"  - {improvement}")

    except Exception as e:
        print(f"[!] Error during evaluation: {e}")
        # Store error in evaluation_feedback
        while len(state["evaluation_feedback"]) <= idx:
            state["evaluation_feedback"].append({})
        state["evaluation_feedback"][idx] = {"error": "Failed to generate evaluation."}
        print("\n--- 📝 FEEDBACK ---\nSorry, an error occurred while generating feedback.")

    # Ask user if they want to see an ideal response
    choice = input("\nDo you want to see an ideal response? (yes/no): ").strip().lower()
    state["wants_ideal"] = True if choice in ("yes", "y") else False

    return state


def generate_answer_node(state: GraphState) -> GraphState:
    """
    Generate ideal answer and advance to next question.
    """
    print("\n--- ✨ GENERATING IDEAL ANSWER ---")
    idx = state.get("current_q_index", 0)
    qas = state.get("generated_qas", [])

    # Guard
    if idx >= len(qas):
        print("[!] generate_answer_node: current_q_index out of range.")
        return state

    question = qas[idx].get("question", "")
    user_answer = qas[idx].get("answer", "")
    
    # Get the feedback for this question
    eval_feedback = state.get("evaluation_feedback", [])
    feedback_text = ""
    if idx < len(eval_feedback) and eval_feedback[idx]:
        fb = eval_feedback[idx]
        if "error" not in fb:
            feedback_text = f"Score: {fb.get('overall_score', 'N/A')}/10. {fb.get('score_justification', '')}"
        
    job_context = state.get("job_role_context", "")

    # Build prompt
    prompt = f"""
    You are an expert interviewer & answer coach.
    Question: {question}
    User's answer: {user_answer}
    Evaluation feedback: {feedback_text}
    Job context: {job_context}

    Generate a concise, well-structured IDEAL answer to the question. Return only the answer text.
    """

    try:
        response = generator_llm.invoke([HumanMessage(content=prompt)])
        ideal_answer = getattr(response, "content", str(response)).strip()
    except Exception as e:
        ideal_answer = f"[Generation failed: {e}]"

    # Ensure ideal_answers list exists and matches length of qas
    ideal_answers = state.get("ideal_answers", [])
    if len(ideal_answers) < len(qas):
        # initialize/extend to match number of questions
        ideal_answers = [""] * len(qas)

    ideal_answers[idx] = ideal_answer
    state["ideal_answers"] = ideal_answers

    # Print the generated ideal answer for user
    print(f"\n--- 🔎 IDEAL ANSWER for Q{idx+1} ---\n{ideal_answer}\n")

    # Reset wants_ideal and advance to next question
    state["wants_ideal"] = False
    state["current_q_index"] = idx + 1

    return state


def decide_next(state: GraphState) -> str:
    """
    Conditional edge decision function.
    """
    # if user requested ideal answer, go there
    if state.get("wants_ideal", False):
        return "generate_answer"

    # if we've finished all questions, end
    if state.get("current_q_index", 0) >= len(state.get("generated_qas", [])):
        return END

    # otherwise, continue interviewing
    return "interview_chat"


def advance_question(state: GraphState) -> GraphState:
    """
    Simple node to advance to next question when user doesn't want ideal answer.
    """
    idx = state.get("current_q_index", 0)
    state["current_q_index"] = idx + 1
    return state


def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("interview_chat", interview_chat)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("advance_question", advance_question)

    # Linear setup: START → research → generator → interview_chat
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "generator")
    workflow.add_edge("generator", "interview_chat")

    # Flow: interview_chat → evaluator → conditional next step
    workflow.add_edge("interview_chat", "evaluator")

    # Conditional edge: evaluator → generate_answer / advance_question
    workflow.add_conditional_edges(
        "evaluator",
        decide_next,
        {
            "generate_answer": "generate_answer",
            "interview_chat": "advance_question",  # Go to advance_question first
            END: END,
        },
    )

    # After advancing question → back to interview_chat
    workflow.add_edge("advance_question", "interview_chat")

    # After generating ideal answer → back to interview_chat
    workflow.add_edge("generate_answer", "interview_chat")

    # Compile
    return workflow.compile()


# Test the graph
app = build_graph()

# Example initial state
init_state: GraphState = {
    "job_role": "Data Scientist",
    "job_role_context": "",
    "candidate_data": [],
    "generated_qas": [],
    "evaluation_feedback": [],
    "current_q_index": 0,
    "wants_ideal": False,
    "ideal_answers": [],
    "num_steps": 0,
    "max_iterations": 10,
}

# Run the graph
if __name__ == "__main__":
    final_state = app.invoke(init_state)
    print("\n--- FINAL STATE ---")
    print("current_q_index:", final_state["current_q_index"])
    print("Number of questions answered:", len([qa for qa in final_state["generated_qas"] if qa.get("answer", "")]))
    print("Number of evaluations:", len(final_state["evaluation_feedback"]))