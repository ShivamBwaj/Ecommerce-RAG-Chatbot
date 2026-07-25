import streamlit as st
import requests
from chatbot_ui.core.config import config
import uuid

st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id=get_session_id()

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

## Lets create a sidebar with a dropdown for the model list and providers

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "used_context" not in st.session_state:
    st.session_state.used_context=[]

with st.sidebar:
    # Create tabs in the sidebar
    suggestions_tab, = st.tabs(["Suggestions"])

    # Suggestions Tab
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item['image_url'], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ok, response_data = api_call(
            "post",
            f"{config.API_URL}/rag/",
            json={"query":prompt,"thread_id":session_id
            },
        )
        
        if ok:
            answer = response_data.get("answer", "I couldn't generate a response this time.")
            used_context=response_data.get("used_context","Context was not found")
            st.session_state.used_context=used_context

        else:
            answer = response_data.get("detail") or response_data.get("answer") or "Request failed"
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content":answer }) 
    st.rerun()
