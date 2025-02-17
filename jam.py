import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Initialize global model and tokenizer
device = "cpu"
try:
    tokenizer = T5Tokenizer.from_pretrained("saved_model")
    model = T5ForConditionalGeneration.from_pretrained("saved_model").to(device)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model. Ensure model is saved correctly.")

# Define request model
class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7
    num_beams: Optional[int] = 5

def generate_answer(model, tokenizer, question, device, max_length=50, temperature=0.7, num_beams=5):
    model.eval()
    input_str = f"question: {question}"
    inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)
    
    outputs = model.generate(
         inputs,
         max_length=max_length,
         num_beams=num_beams,
         do_sample=True,
         temperature=temperature,
         early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.post("/api/generate")
async def generate(request: QuestionRequest):
    try:
        answer = generate_answer(
            model,
            tokenizer,
            request.question,
            device,
            max_length=request.max_length,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 