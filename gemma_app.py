import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="🦙💬 Gemma Model Text Generation with Streamlit")

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer and model
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).to("cuda")
    return tokenizer, model

tokenizer, model = get_tokenizer_model()

def main():
    st.title('Gemma Model Text Generation')

    # Create a Sidebar
    with st.sidebar:
        st.title("Configuration")
        # User input for system prompt
        system_prompt = st.text_input("Enter System Prompt", value="You are a helpful assistant.")
        max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=8192, value=150)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
        top_p = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.9)

    # Initialize chat messages session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    def generate_response(input_text, system_prompt):
        # Prepare input for the model
        prompt = f"{system_prompt}\nUser: {input_text}\nAssistant:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        # Generate text using the model
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode the generated tokens into text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the part of the response after "Assistant:"
        response_start = output_text.find("Assistant:") + len("Assistant:")
        output_text = output_text[response_start:].strip()

        return output_text

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for a chat prompt
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate Gemma response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(user_input, system_prompt)
                    st.write(response)
            # Store the assistant's response in the chat messages
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Button to clear chat history
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if __name__ == '__main__':
    main()
