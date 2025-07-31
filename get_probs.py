import os
import math
from openai import OpenAI

# --- Configuration ---
# Initialize the OpenAI client.
# It's best practice to use an environment variable for your API key.
# You can set it in your terminal like this:
# export OPENAI_API_KEY='your-api-key-here'
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    client = None

# --- Core Function ---

def translate_and_get_probs(
    text_to_translate: str,
    target_language: str = "French",
    source_language: str = "English"
) -> dict:
    """
    Translates a given text using an LLM and returns the translation along
    with the probability of each generated token.

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
        print("OpenAI client is not initialized. Cannot make API call.")
        return None

    # Constructing the prompt for the LLM
    prompt = f"Translate the following {source_language} text to {target_language}: \"{text_to_translate}\""

    try:
        # --- API Call ---
        # We make a call to the chat completions endpoint.
        # Key parameters:
        # - model: We use "gpt-3.5-turbo" as a cost-effective and capable model.
        # - messages: The standard format for chat models.
        # - temperature: Set to 0 for deterministic, high-quality translation.
        # - max_tokens: Limits the length of the translation.
        # - top_p: Set to 1.0.
        # - logprobs: This is the crucial parameter. Setting it to True tells
        #             the API to return the log probabilities of the output tokens.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=256,
            top_p=1.0,
            logprobs=True  # Request token log probabilities
        )

        # --- Response Processing ---
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
    if not client:
        print("\nCannot run example because OpenAI client is not initialized.")
    else:
        # 1. Define the input text
        english_text = "The cat sat on the mat."

        print(f"Original English Text:\n\"{english_text}\"\n")

        # 2. Call the function
        translation_result = translate_and_get_probs(english_text)

        # 3. Print the results
        if translation_result:
            print(f"Translated French Text:\n\"{translation_result['translation_text']}\"\n")
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