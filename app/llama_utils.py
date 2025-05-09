from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-2-7b-chat-hf"  # Bisa diganti yang lebih kecil di awal

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(context, question):
    prompt = f"""Konteks: {context}
    Pertanyaan: {question}
    Jawaban:"""
    result = llama_pipeline(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
    return result.split("Jawaban:")[-1].strip()

