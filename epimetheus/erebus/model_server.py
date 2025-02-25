from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI()


# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt_argocd_model")
model = GPT2LMHeadModel.from_pretrained("./gpt_argocd_model")

class LogRequest(BaseModel):
    log: str

@app.post("/predict")
def predict(request: LogRequest):
    # Encode the incoming log text
    input_ids = tokenizer.encode(request.log, return_tensors="pt")
    # Generate text (we can adjust parameters as needed)
    outputs = model.generate(input_ids, max_lenth=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

# to run, use: uvicorn model_server:app --host 0.0.0.0 --port 80