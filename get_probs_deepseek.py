import os
import math
# The OpenAI library is used to interact with the DeepSeek API
# as DeepSeek maintains compatibility with it.
from openai import OpenAI

# --- Configuration ---
# Initialize the DeepSeek-compatible client.
# You'll need to get an API key from the DeepSeek Platform.
# It's best practice to use an environment variable for your API key.
# You can set it in your terminal like this:
# export DEEPSEEK_API_KEY='your-api-key-here'
try:
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
except TypeError:
    print("DeepSeek API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
    client = None

# --- Core Function ---

def translate_and_get_probs(
    text_to_translate: str,
    target_language: str = "French",
    source_language: str = "English"
) -> dict:
    """
    Translates a given text using the DeepSeek API and returns the translation
    along with the probability of each generated token.

    Args:
        text_to_translate: The string of text to be translated.
        target_language: The language to translate the text into.
        source_language: The original language of the text.

    Returns:
        A dictionary containing:
        - 'translation_text': The translated string.
        - 'token_probabilities': A list of dictionaries, where each dictionary
                                 contains a 'token' and its 'probability'.
        Returns None if the API call fails.
    """
    if not client:
        print("DeepSeek client is not initialized. Cannot make API call.")
        return None

    # Constructing the prompt for the LLM
    prompt = f"Translate the following {source_language} text to {target_language}: \"{text_to_translate}\""

    try:
        # --- API Call ---
        # We make a call to the chat completions endpoint.
        # Key parameters:
        # - model: We use "deepseek-chat", a powerful and general model from DeepSeek.
        # - messages: The standard format for chat models.
        # - temperature: Set to 0 for deterministic, high-quality translation.
        # - logprobs: This is the crucial parameter. Setting it to True tells
        #             the API to return the log probabilities of the output tokens.
        response = client.chat.completions.create(
            model="deepseek-chat", # Using a DeepSeek model
            messages=[
                {"role": "system", "content": f"You are an expert translator. Your task is to translate text from {source_language} to {target_language}. You must return the translated text, without any conversational filler. After the trasnlation, only indicate a list of the spans in the translated text that you were not sure about, no explanations."},
                {"role": "user", "content": text_to_translate}
            ],
            temperature=1.5,  
            max_tokens=256,
            top_p=1.0,
            logprobs=True  # Request token log probabilities
        )

        # --- Response Processing ---
        # The response structure is compatible with OpenAI's
        choice = response.choices[0]
        translated_text = choice.message.content.strip()
        logprobs_content = choice.logprobs.content

        # The API returns log probabilities (log(p)). To get the actual
        # probability (p), we need to calculate e^(log(p)).
        token_probabilities = [
            {
                "token": item.token,
                "probability": math.exp(item.logprob)
            }
            for item in logprobs_content
        ]

        return {
            "translation_text": translated_text,
            "token_probabilities": token_probabilities
        }

    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    print("DeepSeek API Example: Translation with Token Probabilities")
    if not client:
        print("\nCannot run example because DeepSeek client is not initialized.")
    else:
        # 1. Define the input text
        english_text = "What is your name, sir?"

        print(f"--- Using DeepSeek API ---")
        print(f"Original English Text:\n\"{english_text}\"\n")
        source_language = "English"
        target_language = "Quechua"

        # 2. Call the function
        translation_result = translate_and_get_probs(english_text, target_language=target_language, source_language=source_language)

        # 3. Print the results
        if translation_result:
            print(f"Translated {target_language} Text:\n\"{translation_result['translation_text']}\"\n")
            print("--- Token Probabilities ---")
            print(f"{'Token':<15} | {'Probability':<20}")
            print("-" * 38)
            total_prob_product = 1.0
            for item in translation_result['token_probabilities']:
                token = item['token']
                prob = item['probability']
                total_prob_product *= prob
                print(f"{token:<15} | {prob:.4f} ({prob:.2%})")

            print("-" * 38)
            # The overall probability of this exact sequence is the product of individual token probabilities
            print(f"Overall sequence probability: {total_prob_product:.6f}")
