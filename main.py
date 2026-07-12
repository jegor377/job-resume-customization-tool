import uuid
import copy
import os

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict

from resume_generator.generation import generate_resume as gen_resume
from resume_generator.schemas import ArtifactsConfig, ResumeConfig
from schemas import JobRequirements, UserProfile, Experience

st.set_page_config(layout="wide", page_title="Resume customization tool")

artifacts_config = ArtifactsConfig()
llm = ChatOllama(
    model="llama3.2:3b",
    validate_model_on_init=True,
    num_ctx=1024 * 8
)

USER_PROFILE_PATH = "user.json"


if not os.path.exists(artifacts_config.output_dir):
    os.mkdir(artifacts_config.output_dir)
    with open(os.path.join(artifacts_config.output_dir, '.gitignore'), 'w') as f:
        f.write('*')


if not os.path.exists(artifacts_config.dump_dir):
    os.mkdir(artifacts_config.dump_dir)
    with open(os.path.join(artifacts_config.dump_dir, '.gitignore'), 'w') as f:
        f.write('*')


# ---------------------------------------------------------------------------
# Local helper schemas (graph/UI internals, not domain models -> not in schemas.py)
# ---------------------------------------------------------------------------

class UserProfileChoice(BaseModel):
    action: Literal["confirm", "edit"]


class MenuChoice(BaseModel):
    action: Literal["generate", "show_profile", "edit_profile"]


class ReviewChoice(BaseModel):
    action: Literal["accept", "regenerate"]


class EditInstructionType(BaseModel):
    type: Literal["intent", "description"] = Field(
        description=(
            "intent - User just said that they want to edit something but didn't specify what.\n"
            "description - User has provided the description of the things they want to change."
        )
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: UserProfile | None
    job_requirements: JobRequirements | None
    resume_config: ResumeConfig | None
    pdf_path: str | None
    last_choice: str | None
    last_raw_input: str | None
    abort: bool
    ready_to_generate: bool
    edit_intention: str | None


INIT_STATE = {
    "messages": [],
    "user_profile": None,
    "job_requirements": None,
    "resume_config": None,
    "pdf_path": None,
    "last_choice": None,
    "last_raw_input": None,
    "abort": False,
    "ready_to_generate": False,
    "edit_intention": None,
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def bootstrap(state: State):
    """Runs once at the start of every fresh graph run (never after an interrupt)."""
    if state.get("user_profile") is not None:
        return {}
    try:
        with open(USER_PROFILE_PATH) as f:
            return {"user_profile": UserProfile.model_validate_json(f.read())}
    except FileNotFoundError:
        return {}


def bootstrap_router(state: State):
    return "gather_user_profile" if state.get("user_profile") is None else "main_menu"


def gather_user_profile(state: State):
    description = interrupt(
        "Hi! I'm a job resume customization assistant. "
        "Please, tell me about yourself - your commercial experience and skills, "
        "so that I can build your profile."
    )
    structured_llm = llm.with_structured_output(UserProfile)
    profile = structured_llm.invoke([
        SystemMessage(content=(
            "Read the user's description and extract their profile. "
            "If something can't be inferred, leave it None/empty. Don't invent facts."
        )),
        HumanMessage(content=description),
    ])
    save_user_profile(profile)
    return {
        "user_profile": profile,
        "messages": [
            HumanMessage(content=description),
            AIMessage(
                content=(
                    "I've built your profile:\n\n" +
                    pprint_user_profile(profile)
                )
            ),
        ],
    }


def confirm_user_profile(state: State):
    raw = interrupt(
        "Are you satisfied with it? Do you want to change something? "
        "Know that you can make changes in the later stages too."
    )
    classifier = llm.with_structured_output(UserProfileChoice)
    choice = classifier.invoke([
        SystemMessage(content="Classify the user's message into one of the available actions."),
        HumanMessage(content=raw),
    ])
    
    if choice.action == "confirm":
        return {} # do nothing
    
    # otherwise edit the profile
    
    return {
        "last_choice": "edit_profile",
        "last_raw_input": raw,
    }


def confirm_user_profile_router(state: State):
    return {
        None: "main_menu",
        "edit_profile": "edit_profile_classify"
    }[state["last_choice"]]
        


def main_menu(state: State):
    raw = interrupt(
        "What do you want to do?\n"
        "- generate a job resume for a specific job offer\n"
        "- show me my profile\n"
        "- make changes to my profile\n"
    )
    classifier = llm.with_structured_output(MenuChoice)
    choice = classifier.invoke([
        SystemMessage(content="Classify the user's message into one of the available actions."),
        HumanMessage(content=raw),
    ])
    return {
        "messages": [HumanMessage(content=raw)],
        "last_choice": choice.action,
        "last_raw_input": raw,
        "abort": False,
    }


def main_menu_router(state: State):
    return {
        "generate": "gather_job_offer",
        "show_profile": "show_profile",
        "edit_profile": "edit_profile_classify",
    }[state["last_choice"]]


def show_profile(state: State):
    profile = state["user_profile"]
    return {
        "messages": [
            AIMessage(
                content=(
                    "Your current profile:\n\n" +
                    pprint_user_profile(profile)
                )
            )
        ],
    }


def edit_profile_classify(state: State):
    if state["edit_intention"] is None:
        classifier = llm.with_structured_output(EditInstructionType)
        intention = classifier.invoke([
            SystemMessage(content=(
                "User wants to update their profile configuration. "
                "Classify whether the user provided the change description "
                "(e.g. \"Add X to my skills.\", where X is a specified skill) "
                "or just said that they want to change something "
                "(e.g. I want to edit my profile.)."
            )),
            HumanMessage(content=state.get("last_raw_input") or ""),
        ])
        return {
            "edit_intention": intention,
        }
    
    return {} # do nothing


def edit_profile(state: State):
    print("INTENT:", state["edit_intention"], "INSTR:", state.get("last_raw_input") or "")

    if state["edit_intention"].type == "intent":
        instruction = interrupt(
            "What do you want to change in your profile? Describe the changes. "
            "Say **abort** if you want to go back to the main menu."
        )
    else:
        instruction = state.get('last_raw_input') or ""
    
    if instruction == "abort":
        return {
            "abort": True,
            "messages": [
                HumanMessage(content=instruction),
                AIMessage(content="Aborting and going back to the main menu."),
            ],
            "edit_intention": None,
        }
    
    structured_llm = llm.with_structured_output(UserProfile)
    updated = structured_llm.invoke([
        SystemMessage(content=(
            "Update the existing user profile based on the requested changes. "
            "Keep fields that weren't mentioned unchanged."
        )),
        HumanMessage(content=(
            f"Current profile:\n{state['user_profile'].model_dump_json()}\n\n"
            f"Requested changes:\n{instruction}"
        )),
    ])
    print("CURR PROFILE:", state['user_profile'], "NEW PROFILE:", updated)
    save_user_profile(updated)
    return {
        "user_profile": updated,
        "messages": [
            AIMessage(
                content=(
                    f"I've updated your profile:\n\n" +
                    pprint_user_profile(updated)
                )
            )
        ],
        "edit_intention": None,
    }


def gather_job_offer(state: State):
    offer_text = interrupt(
        "Paste the content of the job offer (role, project description, "
        "must have and nice to have requirement). If you want to go back, please say: **abort**."
    )
    if offer_text == 'abort':
        return {
            "abort": True,
            "messages": [
                HumanMessage(content=offer_text),
                AIMessage(content="Aborting and going back to the main menu."),
            ],
        }
    structured_llm = llm.with_structured_output(JobRequirements)
    requirements = structured_llm.invoke([
        SystemMessage(content=(
            "Extract structured job requirements from the pasted job offer. "
            "Separate must-have from nice-to-have skills. "
            "Make sure to add expected technologies as must-have skills."
        )),
        HumanMessage(content=offer_text),
    ])
    return {
        "abort": False,
        "job_requirements": requirements,
        "messages": [
            HumanMessage(content=offer_text),
            AIMessage(content=f"I've understood the offer:\n\n```json {requirements.model_dump_json(indent=2)} ```"),
        ],
    }


def gather_job_offer_router(state: State):
    return "main_menu" if state["abort"] else "confirm_job_offer_info"


def confirm_job_offer_info(state: State):
    intent = interrupt(
        "Are you satisfied with it? Do you want to change something? "
        "Say **confirm** if you are satisfied. If else then say what to change."
    )
    if intent.lower() == 'confirm':
        return {
            "messages": [
                HumanMessage(content=intent),
            ],
            "ready_to_generate": True,
        }
    
    return {
        "messages": [
            HumanMessage(content=intent),
        ],
        "ready_to_generate": False,
        "last_raw_input": intent,
    }
    

def confirm_job_offer_info_router(state: State):
    return "build_resume_config" if state["ready_to_generate"] else "edit_job_requirements_classification"


def edit_job_requirements_classification(state: State):
    if state["edit_intention"] is None:
        classifier = llm.with_structured_output(EditInstructionType)
        intention = classifier.invoke([
            SystemMessage(content=(
                "User wants to update the job requirements configuration. "
                "Classify whether the user provided the change description "
                "(e.g. \"Change role to X.\", where X is a specified role) "
                "or just said that they want to change something "
                "(e.g. I want to edit the job requirements.)."
            )),
            HumanMessage(content=state.get("last_raw_input") or ""),
        ])
        return {
            "edit_intention": intention.type
        }
    return {} # do nothing


def edit_job_requirements(state: State):
    if state["edit_intention"] == "intent":
        instruction = interrupt(
            "What do you want to change in the infered job requirements? "
            "Describe the changes. Say **abort** if you want to go back "
            "to the main menu."
        )
    else:
        instruction = state.get('last_raw_input') or ""
    
    if instruction == "abort":
        return {
            "abort": True,
            "messages": [
                HumanMessage(content=instruction),
                AIMessage(content="Aborting and going back to the main menu."),
            ],
            "edit_intention": None,
        }
    
    structured_llm = llm.with_structured_output(JobRequirements)
    updated = structured_llm.invoke([
        SystemMessage(content=(
            "Update the existing job requirements based on the requested changes. "
            "Keep fields that weren't mentioned unchanged."
        )),
        HumanMessage(content=(
            f"Current job requirements:\n{state['job_requirements'].model_dump_json()}\n\n"
            f"Requested changes:\n{instruction}"
        )),
    ])

    return {
        "job_requirements": updated,
        "messages": [
            AIMessage(
                content=(
                    "I've updated the job requirements:\n\n"
                    f"```json {updated.model_dump_json(indent=2)} ```"
                )
            )
        ],
        "edit_intention": None,
    }


def edit_job_requirements_router(state: State):
    return "main_menu" if state["abort"] else "confirm_job_offer_info"


def build_resume_config(state: State):
    possible_languages = [
        name.replace('template_', '').replace('.tex', '')
        for name in os.listdir(artifacts_config.templates_path)
    ]
    language = interrupt(
        "Specify the job resume language that you want to use. " +
        "Possible values are:\n" +
        '\n'.join([f'- {name}' for name in possible_languages])
    )
    structured_llm = llm.with_structured_output(ResumeConfig)
    resume_config = structured_llm.invoke([
        SystemMessage(content=(
            "Compare the candidate's profile with the job requirements. "
            "Produce a ResumeConfig tailored to this specific offer: write about_me "
            "to match the role, group relevant strengths into named sections, and "
            "for each experience entry keep only the accomplishments relevant to this offer."
        )),
        HumanMessage(content=(
            f"Candidate profile:\n{state['user_profile'].model_dump_json()}\n\n"
            f"Job requirements:\n{state['job_requirements'].model_dump_json()}"
        )),
    ])
    
    if language != 'en':
        structured_llm = llm.with_structured_output(ResumeConfig)
        resume_config = structured_llm.invoke([
            SystemMessage(content=(
                "Translate the resume config to the specified language. "
                "Language is specified by it's code name.\n"
                "- en - an English language.\n"
                "- pl - a Polish language."
            )),
            HumanMessage(content=(
                f"Language code name: {language}\n\n"
                f"Resume config:\n{resume_config.model_dump_json()}"
            )),
        ])
    
    filename = state["job_requirements"].company_name + "_" + language
    
    try:
        pdf_path = gen_resume(
            resume_config=resume_config,
            artifacts_conf=artifacts_config,
            language=language,
            filename=filename,
        )
    except SystemError:
        return {
            "resume_config": resume_config,
            "messages": [
                AIMessage(
                    content=(
                        "Error while generating the job resume. "
                        "Please, try again later."
                    )
                )
            ]
        }

    return {
        "resume_config": resume_config,
        "pdf_path": pdf_path,
        "messages": [AIMessage(content="I've generated a tailored resume configuration and file.")],
    }
    

def build_resume_config_router(state: State):
    return "main_menu" if state["pdf_path"] is None else "review_config"


def review_config(state: State):
    raw = interrupt(
        "Do you accept the generated job resume? Answer e.g. \"accept\" or \"change **X** because...\"."
    )
    classifier = llm.with_structured_output(ReviewChoice)
    choice = classifier.invoke([
        SystemMessage(content="Classify the user's decision about the generated resume."),
        HumanMessage(content=raw),
    ])
    return {
        "messages": [HumanMessage(content=raw)],
        "last_choice": choice.action,
    }


def review_config_router(state: State):
    return "main_menu" if state["last_choice"] == "accept" else "build_resume_config"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("bootstrap", bootstrap)
graph_builder.add_node("gather_user_profile", gather_user_profile)
graph_builder.add_node("confirm_user_profile", confirm_user_profile)
graph_builder.add_node("main_menu", main_menu)
graph_builder.add_node("show_profile", show_profile)
graph_builder.add_node("edit_profile_classify", edit_profile_classify)
graph_builder.add_node("edit_profile", edit_profile)
graph_builder.add_node("gather_job_offer", gather_job_offer)
graph_builder.add_node("confirm_job_offer_info", confirm_job_offer_info)
graph_builder.add_node("edit_job_requirements_classification", edit_job_requirements_classification)
graph_builder.add_node("edit_job_requirements", edit_job_requirements)
graph_builder.add_node("build_resume_config", build_resume_config)
graph_builder.add_node("review_config", review_config)

graph_builder.add_edge(START, "bootstrap")
graph_builder.add_conditional_edges(
    "bootstrap", bootstrap_router, ["gather_user_profile", "main_menu"]
)
graph_builder.add_edge("gather_user_profile", "confirm_user_profile")
graph_builder.add_conditional_edges(
    "confirm_user_profile",
    confirm_user_profile_router,
    ["main_menu", "edit_profile_classify"],
)
graph_builder.add_conditional_edges(
    "main_menu",
    main_menu_router,
    ["gather_job_offer", "show_profile", "edit_profile_classify"],
)
graph_builder.add_edge("edit_profile_classify", "edit_profile")
graph_builder.add_edge("show_profile", "main_menu")
graph_builder.add_edge("edit_profile", "main_menu")
graph_builder.add_conditional_edges(
    "gather_job_offer",
    gather_job_offer_router,
    ["main_menu", "confirm_job_offer_info"],
)
graph_builder.add_conditional_edges(
    "confirm_job_offer_info",
    confirm_job_offer_info_router,
    ["build_resume_config", "edit_job_requirements_classification"],
)
graph_builder.add_edge("edit_job_requirements_classification", "edit_job_requirements")
graph_builder.add_conditional_edges(
    "edit_job_requirements",
    edit_job_requirements_router,
    ["main_menu", "confirm_job_offer_info"],
)
graph_builder.add_conditional_edges(
    "build_resume_config",
    build_resume_config_router,
    ["main_menu", "review_config"],
)
graph_builder.add_conditional_edges(
    "review_config",
    review_config_router,
    ["build_resume_config", "main_menu"],
)


# ---------------------------------------------------------------------------
# Streamlit driver
# ---------------------------------------------------------------------------

def get_pending_interrupt(config):
    """Returns the payload of the interrupt the graph is currently paused on, or None."""
    snapshot = st.session_state.graph.get_state(config)
    if snapshot.next:
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "graph" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
    st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)
    st.session_state.graph.invoke(
        copy.deepcopy(INIT_STATE),
        config=config,
    )

state = st.session_state.graph.get_state(config)
messages = state.values.get("messages", [])

col1, col2 = st.columns([1, 1])

with col1:
    chat_container = st.container(height=350)
    with chat_container:
        for message in messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

        pending = get_pending_interrupt(config)
        if pending:
            with st.chat_message("assistant"):
                st.markdown(pending)

with col2:
    st.write("Generated job resume")
    pdf_path = state.values.get("pdf_path")
    if pdf_path:
        st.pdf(pdf_path)

if prompt := st.chat_input("Your message"):
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if get_pending_interrupt(config) is not None:
                    st.session_state.graph.invoke(Command(resume=prompt), config=config)
                else:
                    # graf zakończył działanie (np. user wybrał "zakończ")
                    # zaczynamy nowy przebieg na tym samym wątku
                    st.session_state.graph.invoke(
                        copy.deepcopy(INIT_STATE),
                        config=config,
                    )
    st.rerun()