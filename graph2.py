from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict

from resume_generator.generation import generate_resume as gen_resume
from resume_generator.schemas import ArtifactsConfig, ResumeConfig
from schemas import JobRequirements, UserProfile, Experience

artifacts_config = ArtifactsConfig()
llm = ChatOllama(
    model="llama3.2:3b",
    validate_model_on_init=True,
    num_ctx=1024 * 8,
)

USER_PROFILE_PATH = "user.json"

# classes


class UserIntent(BaseModel):
    intent: Literal[
        'chat',
        'show_user_profile',
        'generate_job_resume'
    ] = Field(
        'chat',
        description="Possible user intents. Anything but chat is a command."
    )

class UserProfileFeedback(BaseModel):
    grade: Literal[
        "correct",
        "incorrect"
    ] = Field(
        description="Decide if the infered user profile update is correct or contains issues (incorrect)."
    )
    feedback: str = Field(
        description="If the infered user profile is incorrect, provide feedback on how to improve it."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    init_message: bool
    user_intent: Literal[
        'init_message',
        'chat',
        'show_user_profile',
        'generate_job_resume'
    ] | None
    user_msg: str | None
    user_profile: UserProfile | None
    summarized_chat: str | None
    grade: str | None
    feedback: str | None


INIT_STATE = {
    "messages": [],
    "user_intent": None,
    "user_msg": None,
    "user_profile": None,
    "summarized_chat": None,
    "grade": None,
    "feedback": None,
}


def save_user_profile(profile: UserProfile) -> None:
    with open(USER_PROFILE_PATH, "w") as f:
        f.write(profile.model_dump_json(indent=2))


def pprint_user_profile(user_profile: UserProfile):
    tab = '    '
    def get_accomplishments(experience: Experience):
        return '\n'.join([
            f"{tab}- {accomplishment}"
            for accomplishment in experience.accomplishments
        ])
    def get_used_technologies(experience: Experience):
        return '\n'.join([
            f"{tab}- {used_technology}"
            for used_technology in experience.used_technologies
        ])
    def get_responsibilities(experience: Experience):
        return '\n'.join([
            f"{tab}- {responsibility}"
            for responsibility in experience.responsibilities
        ])
        
    experiences = "\n".join([
        (
            f"#### Company: {experience.company}\n" +
            f"- role: {experience.role}\n" +
            f"- start: {experience.start}\n" +
            f"- end: {experience.end}\n" +
            "- accomplishments:\n" +
            get_accomplishments(experience) + "\n"
            "- used technologies:\n" +
            get_used_technologies(experience) + "\n"
            "- responsibilities:\n" +
            get_responsibilities(experience) + "\n"
        )
        for experience in user_profile.experiences
    ])
    
    skills = '\n'.join([f"- {skill}" for skill in user_profile.skills])
    
    return (
        "### Summary\n"
        f"{user_profile.summary}\n"
        "### Experiences\n"
        f"{experiences}"
        "### Skills\n"
        f"{skills}"
    )


# nodes

def initial(state: State):
    if state["init_message"]:
        return {}
    return {
        "messages": [
            AIMessage(
                content="Hi! I'm a job resume customization assistant."
            )
        ],
    }
   

def intent_router(state: State):
    classifier = llm.with_structured_output(UserIntent)
    if len(state["messages"]) == 0:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Hi! I'm a job resume customization assistant. "
                        "Tell me about yourself. I'll be gathering your "
                        "information to create your user profile. At any "
                        "point you can command me to show this profile or "
                        "to create a job resume."
                    )
                )
            ],
            "user_intent": "init_message",
        }
    user_msg = state["messages"][-1].content
    user_intent = classifier.invoke([
        SystemMessage(
            content=(
                "Classify user message intent. "
                "If the user didn't issue any command, then assume that the user wants to chat."
            )
        ),
        HumanMessage(content=user_msg),
    ])
    return {
        "user_intent": user_intent.intent,
        "user_msg": user_msg,
    }


def intent_decision(state: State):
    return {
        'init_message': END,
        'chat': 'gather_chat_info',
        'show_user_profile': 'chat',
        'show_profile': 'show_profile',
        'generate_job_resume': 'generate_job_resume',
    }[state["user_intent"]]


def gather_chat_info(state: State):
    structured_llm = llm.with_structured_output(UserProfile)
    feedback_msg = state["feedback"] or "(none)"
    system_message = (
        "You are a job resume assistant. Your task is to gather "
        "information about the user.\n"
        "Rules:\n"
        "- Newly generated user profile should differ from previously "
        "generated only by information in last user message.\n"
        "- The fields that are not mentioned by the user should be the "
        "same as in previously generated user profile.\n"
        "- No new facts should be invented. Only facts from last user "
        "message should be included.\n"
        "- Ignore anything that isn't related to job experiences, "
        "skills, and professional background.\n"
        "- It's very important that you take evaluator feedback into "
        "account whenever present. Fix any issues from evaluator feedback.\n\n"
        "Evaluator feedback:\n{feedback_msg}\n\n"
    )
    
    print(f"Feedback:\n{feedback_msg}")
    
    if state["user_profile"]:
        user_profile: UserProfile = structured_llm.invoke([
            SystemMessage(
                content=(
                    f"{system_message}\n\n"
                    f"{feedback_msg}"
                    f"Current profile:\n{state['user_profile'].model_dump_json()}"
                )
            ),
            HumanMessage(content=state["user_msg"]),
        ])
    else:
        user_profile: UserProfile = structured_llm.invoke([
            SystemMessage(
                content=f"{system_message}\n\n{feedback_msg}"
            ),
            HumanMessage(content=state["user_msg"]),
        ])
    
    # remove empty companies
    user_profile.experiences = [
        experience
        for experience in user_profile.experiences
        if experience.company.strip().casefold() != "" 
    ]
    
    return {
        "user_profile": user_profile,
        "feedback": None,
    }


def evaluate_generated_user_profile(state: State):
    evaluator = llm.with_structured_output(UserProfileFeedback)
    schema = UserProfile.model_json_schema()
    message = (
        "Your job is to only grade the newly generated user profile "
        "based on last user message and provide feedback if necessary. "
        "Repeating company names aren't allowed.\n\n"
        f"User profile schema: {schema}\n\n"
        f"Newly generated user profile:\n{state["user_profile"].model_dump_json()}\n\n"
        f"Last user message:\n{state["user_msg"]}"
    )
    grade: UserProfileFeedback = evaluator.invoke(message)
    print(f"EVALUATION:\n\nGrade: {grade.grade}\nFeedback: {grade.feedback}")
    return {
        "grade": grade.grade,
        "feedback": grade.feedback,
    }
    

def route_user_profile(state: State):
    return "regenerate" if state["grade"] == "incorrect" else "pass"


def chat(state: State):
    print(f"Current user profile:\n{state["user_profile"].model_dump_json(indent=2)}")
    print(f"Current chat summary:\n{state['summarized_chat']}")
    prompt = (
        "You are a job resume assistant. "
        "Your task is to gather information about the user. "
        "Whenever possible, try to steer the conversation towards talking about "
        "job experiences, skills, and professional background. "
        "You can look at the current profile, which is summarized structured "
        "information about the user. Be concise and professional.\n\n"
        f"Current profile:\n{state["user_profile"].model_dump_json()}\n\n"
        f"Previous conversation summarization:\n{state["summarized_chat"] or '(none)'}"
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_msg"]),
    ])
    
    summarized_chat = llm.invoke([
        SystemMessage(
            content=(
                "Create a short memory of this conversation. "
                "Always write something. "
                "Include user facts, goals, preferences, and discussed topics. "
                "If there are no facts yet, summarize the current topic. "
                "Keep it short, preferably a few lines of text.\n\n"
                f"Previous summarization:\n{state["summarized_chat"] or '(none)'}"
            )
        ),
        HumanMessage(
            content=(
                f"User message:\n{state['user_msg']}\n\n"
                f"Assistant response:\n{response.content}"
            )
        ),
    ])
    
    print(f"New chat summary:\n{summarized_chat.content}")
    
    return {
        "messages": [
            response
        ],
        "summarized_chat": summarized_chat.content,
    }


def generate_job_resume(state: State):
    return {
        "messages": [
            AIMessage(content="generate_job_resume message")
        ],
    }


def create_workflow() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("intent_router", intent_router)
    workflow.add_node("gather_chat_info", gather_chat_info)
    workflow.add_node("evaluate_generated_user_profile", evaluate_generated_user_profile)
    workflow.add_node("chat", chat)
    workflow.add_node("generate_job_resume", generate_job_resume)

    workflow.add_edge(START, "intent_router")
    workflow.add_conditional_edges(
        "intent_router",
        intent_decision,
        ["gather_chat_info", "chat", "generate_job_resume", END]
    )
    workflow.add_edge("gather_chat_info", "evaluate_generated_user_profile")
    workflow.add_conditional_edges(
        "evaluate_generated_user_profile",
        route_user_profile,
        {
            "regenerate": "gather_chat_info",
            "pass": "chat",
        }
    )
    workflow.add_edge("chat", END)
    workflow.add_edge("generate_job_resume", END)

    return workflow
