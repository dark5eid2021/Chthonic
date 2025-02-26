import tiktoken

def estimate_tokens(text):
    """
    Estimate the number of token in a given text using OpenAI's tokenizer

    Cost saving options
        Local tokenization - using libraries like Hugging Face's transformers or tiktoken can reduce costs if you process
            the text on your own hardware
        Batch processing - some api's offer discounts on bulk processing
        Efficient tokenization - using a different tokenizer (like SentencePiece) can yield fewer tokens

    """


    enc = tiktoken.get_encoding("cl100k_base") # OpenAI's GPT-4 tokenizer
    tokens = enc.encode(text)
    return len(tokens)




# example usage
if __name__ == "__main__":
    sample_text = "This is a sample text to estimate token count. " * 1000 # simulate large text
    token_count = estimate_tokens(sample_text)
    print(f"Estimated tokens: {token_count}")