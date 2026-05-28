import base64
import logging
import streamlit as st

_USER_AVATAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#0ea5e9"/>
  <circle cx="50" cy="38" r="18" fill="white"/>
  <ellipse cx="50" cy="90" rx="30" ry="26" fill="white"/>
</svg>"""
USER_AVATAR = "data:image/svg+xml;base64," + base64.b64encode(_USER_AVATAR_SVG.encode()).decode()

from enterprise_ai.agents.orchestrator_caching import OrchestratorAgent

# Show INFO-level logs in terminal for Plan-Action-Reflect debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Page setup
st.set_page_config(
    page_title='Nowacoski',
    page_icon="🫴",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = OrchestratorAgent()
if "is_agent_processing" not in st.session_state:
    st.session_state.is_agent_processing = False

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    # Mode toggle
    mode = st.radio("Mode", ["💬 Chat", "📋 Batch"], horizontal=True)
    st.session_state["app_mode"] = mode
    st.markdown("---")

    st.markdown("### Agent Info")
    with st.expander("App Connections 🙂"):
        st.write("Google Workspace")
        st.write("Tavily")
        st.write("Codebases")

    # !!!for reference!!!
    # with st.expander("Available tools"):
    #     tools = st.session_state.agent.tool_names
    #     for tool_name in tools:
    #         st.write(f"• {tool_name}")

    if mode == "💬 Chat":
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.agent.messages = []
            # Reset conversation memory so follow-up context starts fresh
            from enterprise_ai.agents.orchestrator_caching import ConversationMemory
            st.session_state.agent.conversation_memory = ConversationMemory()
            st.rerun()


# Main content
st.title("Nowacoski")
st.markdown("Your assistant to onboard full-time employees.")

if st.session_state.get("app_mode") == "📋 Batch":
    # ── Batch mode ───────────────────────────────────────────────
    from enterprise_ai.batch_ui import render_batch_upload, render_batch_progress

    if "batch_workflow_id" in st.session_state:
        render_batch_progress()
    else:
        render_batch_upload()

else:
    # ── Chat mode (default) ─────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar="https://ca.slack-edge.com/E08ABN19WSK-U010NJG1P2T-ffb30871fb55-512"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(message["content"])


    # Chat input
    def generate_response_callback():
        user_prompt = st.session_state.chat_input_key
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.is_agent_processing = True

    st.chat_input(
        "Ask Nowacoski...",
        on_submit=generate_response_callback,
        disabled=st.session_state.is_agent_processing,
        key="chat_input_key"
    )

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.is_agent_processing:
        prompt = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant", avatar="https://ca.slack-edge.com/E08ABN19WSK-U010NJG1P2T-ffb30871fb55-512"):
            bot_output = st.empty()
            try:
                bot_output.empty()
                with st.spinner("Processing...this may take a while"):
                    response = st.session_state.agent(prompt)
                bot_output.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                if "credentials" in str(e):
                    error_msg = "Missing IAM permissions for Nowacoski."
                else:
                    error_msg = f"Oh noes! {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            finally:
                st.session_state.is_agent_processing = False
                st.rerun()
