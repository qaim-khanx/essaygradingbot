from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API model
llm = ChatOpenAI(model="gpt-3.5-turbo")  # You can switch to "gpt-4" if you have access

# Define the State class to represent grading status
class State(TypedDict):
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float

# Grading functions
def check_relevance(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevance of the following essay to the given topic. "
        "Provide a relevance score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["relevance_score"] = float(result.content.split(":")[1].strip())
    return state

def check_grammar(state: State) -> State:
    prompt = "Analyze the grammar of this essay. Provide a score between 0 and 1. Essay: {essay}"
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["grammar_score"] = float(result.content.split(":")[1].strip())
    return state

def analyze_structure(state: State) -> State:
    prompt = "Analyze the structure of this essay. Provide a score between 0 and 1. Essay: {essay}"
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["structure_score"] = float(result.content.split(":")[1].strip())
    return state

def evaluate_depth(state: State) -> State:
    prompt = "Evaluate the depth of analysis in this essay. Provide a score between 0 and 1. Essay: {essay}"
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["depth_score"] = float(result.content.split(":")[1].strip())
    return state

def calculate_final_score(state: State) -> State:
    state["final_score"] = (
        state["relevance_score"] * 0.25 +
        state["grammar_score"] * 0.25 +
        state["structure_score"] * 0.25 +
        state["depth_score"] * 0.25
    )
    return state

def grade_essay(essay: str) -> dict:
    initial_state = State(
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0
    )

    initial_state = check_relevance(initial_state)
    initial_state = check_grammar(initial_state)
    initial_state = analyze_structure(initial_state)
    initial_state = evaluate_depth(initial_state)
    initial_state = calculate_final_score(initial_state)

    return initial_state

