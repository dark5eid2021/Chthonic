import os

try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    print("tiktoken not found. Falling back to character-based estimation.")
    USE_TIKTOKEN = False

def estimate_tokens(text, avg_chars_per_token=4):
    """Estimates token count using OpenAI's tokenizer or a simple character-based method."""
    if USE_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer (GPT-4 & GPT-3.5)
        return len(enc.encode(text))
    else:
        return len(text) // avg_chars_per_token  # Fallback estimation

def process_text_file(file_path):
    """Reads a text file and estimates token count."""
    if not os.path.exists(file_path):
        print("File not found!")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_count = estimate_tokens(text)
    print(f"Estimated tokens in {file_path}: {token_count}")
    return token_count

# Example usage
file_path = "your_text_file.txt"  # Replace with your actual text file path
process_text_file(file_path)
