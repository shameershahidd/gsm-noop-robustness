import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Models we are testing
MODELS = {
    "llama-3.3-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    "qwen/qwen3-32b": "groq"
}

def call_model(model_name, prompt, max_retries=3):
    """Call a model and return the response text."""
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def run_self_consistency(model_name, prompt, n=5):
    """Run the same prompt n times and return list of responses."""
    responses = []
    for _ in range(n):
        response = call_model(model_name, prompt)
        if response:
            responses.append(response)
        time.sleep(0.5)
    return responses


if __name__ == "__main__":
    test_prompt = "What is 2 + 2? Give only the number."

    print("Testing Llama-3.3-70b...")
    result = call_model("llama-3.3-70b-versatile", test_prompt)
    print(f"Response: {result}")

    print("\nTesting Llama-3.1-8b...")
    result = call_model("llama-3.1-8b-instant", test_prompt)
    print(f"Response: {result}")

    print("\nTesting Qwen3-32b...")
    result = call_model("qwen/qwen3-32b", test_prompt)
    print(f"Response: {result}")