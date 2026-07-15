
def prepare_chat_format(messages, model_name):
    """Format the chat messages according to the model's expected format."""
    model_name_lower = model_name.lower()
    
    # Meta Llama models
    if "llama" in model_name_lower:
        return format_llama_chat(messages)
    # Google Gemma models
    elif "gemma" in model_name_lower:
        return format_gemma_chat(messages)
    # Cohere models
    elif "cohere" in model_name_lower or "aya" in model_name_lower or "c4ai" in model_name_lower:
        return format_cohere_chat(messages)
    # Microsoft Phi models
    elif "phi" in model_name_lower:
        return format_phi_chat(messages)
    # Tower Babel models
    elif "babel" in model_name_lower:
        return format_babel_chat(messages)
    # DeepSeek models
    elif "deepseek" in model_name_lower:
        return format_deepseek_chat(messages)
    # Default/generic format
    else:
        return format_generic_chat(messages)
        
def format_llama_chat(messages):
    """Format for Llama 3.1/3.2/3.3 models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    # Get only user/assistant messages
    chat_messages = [m for m in messages if m["role"] in ["user", "assistant"]]
    
    # Start the conversation with the system message if provided
    if system_message:
        formatted_prompt = f"<|system|>\n{system_message}\n"
    
    # Add the rest of the conversation
    for msg in chat_messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
    
    # Add the final assistant prompt
    if chat_messages[-1]["role"] == "user":
        formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

def format_gemma_chat(messages):
    """Format for Gemma 2/3 models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    if system_message:
        formatted_prompt += f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<start_of_turn>model\n"
    
    return formatted_prompt

def format_cohere_chat(messages):
    """Format for Cohere and Aya models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    if system_message:
        formatted_prompt += f"<chat_message role=\"system\">{system_message}</chat_message>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<chat_message role=\"user\">{msg['content']}</chat_message>\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<chat_message role=\"assistant\">{msg['content']}</chat_message>\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<chat_message role=\"assistant\">"
    
    return formatted_prompt

def format_phi_chat(messages):
    """Format for Microsoft Phi models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    # Add system message if exists
    if system_message:
        formatted_prompt += f"<|system|>\n{system_message}\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

def format_babel_chat(messages):
    """Format for Tower Babel models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    if system_message:
        formatted_prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt

def format_deepseek_chat(messages):
    """Format for DeepSeek models."""
    formatted_prompt = ""
    system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    # Add system message if exists
    if system_message:
        formatted_prompt += f"<|system|>\n{system_message}\n<|endoftext|>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}\n<|endoftext|>\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{msg['content']}\n<|endoftext|>\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

def format_generic_chat(messages):
    """Generic chat format for models not specifically handled."""
    formatted_prompt = ""
    
    for message in messages:
        formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    
    # Add final assistant prompt
    if messages[-1]["role"] == "user":
        formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt
