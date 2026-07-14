import uuid
import copy
import os

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from graph import create_workflow, INIT_STATE, artifacts_config


st.set_page_config(layout="wide", page_title="Resume customization tool")


if not os.path.exists(artifacts_config.output_dir):
    os.mkdir(artifacts_config.output_dir)
    with open(os.path.join(artifacts_config.output_dir, '.gitignore'), 'w') as f:
        f.write('*')


if not os.path.exists(artifacts_config.dump_dir):
    os.mkdir(artifacts_config.dump_dir)
    with open(os.path.join(artifacts_config.dump_dir, '.gitignore'), 'w') as f:
        f.write('*')


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
    st.session_state.graph = create_workflow().compile(checkpointer=st.session_state.checkpointer)
    st.session_state.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
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
                        {
                            "messages": [
                                HumanMessage(content=prompt)
                            ]
                        },
                        config=config,
                    )
    st.rerun()