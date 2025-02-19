import os
import json
import traceback

# Ensure CUDA is disabled
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        torch.set_default_tensor_type(torch.FloatTensor)
        _tokenizer = T5Tokenizer.from_pretrained("saved_model")
        _model = T5ForConditionalGeneration.from_pretrained(
            "saved_model",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        _model.eval()
    return _model, _tokenizer

def generate_answer(question, max_length=50, temperature=0.7, num_beams=5):
    model, tokenizer = load_model()
    input_str = f"question: {question}"
    inputs = tokenizer.encode(input_str, return_tensors="pt")
    import torch
    with torch.no_grad():
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

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        question = body['question']
        max_length = body.get('max_length', 150)
        temperature = body.get('temperature', 0.7)
        num_beams = body.get('num_beams', 5)
        answer = generate_answer(question, max_length, temperature, num_beams)
        return {
            'statusCode': 200,
            'body': json.dumps({'answer': answer}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print("Error occurred:", error_details)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }

# FastAPI app for local testing
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/api/generate")
async def api_generate(request: Request):
    event = {"body": (await request.body()).decode("utf-8")}
    return lambda_handler(event, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jam:app", host="0.0.0.0", port=8000, reload=True)
