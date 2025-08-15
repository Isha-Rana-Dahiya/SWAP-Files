**AI-Powered Scoring for the SWAP Assessment Form**

I have this attached assessment form. It is in english and thai language. it has 20 questions. from question 1-12, first two options give 0 score and last two options give 2 score. for question 13-20, first two options give 1 score and last two options give 3 score. i want that when employees fill this form online, a total score is calculated from question 1-12 as Wellness Score and a total score from question 13-20 is calculated as Insomnia Score. I want that when I attach the filled form and click on generate wellness score and generate insomnia score, I should be able to get the two scores, wellness and insomnia score. I want to use free AI tools available to implement this.  give step by step how can i implement above. consider i am a beginner and i want each step told in detail to be able to implement it easily. I want above to be implemented using AI.

This guide shows you how to build a free, end-to-end pipeline that:
- Lets employees upload a filled PDF of the SWAP form
- Uses an open-source LLM to parse answers in English/Thai
- Computes a Wellness Score (Q1–Q12) and an Insomnia Score (Q13–Q20)
- Displays both scores via a simple web interface
We’ll use Google Colab, pdfplumber, Hugging Face’s free Llama 2 model, and Gradio.

Step 1: Prepare Your Google Colab Notebook
- Open https://colab.research.google.com/ and sign in.
- Click File → New notebook and rename it (e.g. “SWAP AI Scorer)

- In the first cell, install required libraries:
!pip install pdfplumber gradio transformers torch sentencepiece

Run the cell and wait for installation to finish.

Step 2: Extract Text from the Filled PDF
Use pdfplumber to pull all the text (including your checkbox labels) into one string:

import pdfplumber

def extract_text(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_pages.append(page.extract_text() or "")
    return "\n".join(text_pages)


Upload a sample filled form via the Colab sidebar, then test:
Python:
raw = extract_text("filled_swap.pdf")
print(raw[:500])



Step 3: Load a Free Open-Source LLM
We’ll use Meta’s Llama 2-7b-chat-hf, which is free for research/non-commercial use:

Python:
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.1
)

If you run out of RAM, switch to a smaller model such as TheBloke/vicuna-7b-1.1-HF.

Step 4: Prompt the LLM to Assign Numeric Scores
Craft a prompt that:
- Extracts each selected answer
- Applies your scoring rules
- Returns a JSON object with scores for Q1–Q20
  Python:
  
import json, re

def ask_llm_for_scores(raw_text):
    prompt = f"""
You have a filled SWAP assessment in English and Thai.  
Extract the answer for each question 1–20 and map it to a numeric score:  
- Q1–Q12: first two options = 0, last two options = 2  
- Q13–Q20: first two options = 1, last two options = 3  

Output strictly valid JSON like:
{{ "Q1": 2, "Q2": 0, …, "Q20": 3 }}

Here is the form text:
{raw_text}
"""
    resp = chat(prompt)[0]["generated_text"]
    json_str = re.search(r"\{.*\}", resp, re.S).group()
    return json.loads(json_str)


Test it:
Python:
scores = ask_llm_for_scores(raw)
print(scores)



Step 5: Compute Totals and Build the Web UI
Sum the per-question scores into Wellness and Insomnia totals, then wrap everything in Gradio:
Python:
import gradio as gr

def compute_scores(pdf_file):
    raw = extract_text(pdf_file.name)
    scores = ask_llm_for_scores(raw)
    wellness  = sum(scores[f"Q{i}"] for i in range(1, 13))
    insomnia  = sum(scores[f"Q{i}"] for i in range(13, 21))
    return wellness, insomnia

iface = gr.Interface(
    fn=compute_scores,
    inputs=gr.File(label="Upload your filled SWAP PDF"),
    outputs=[gr.Number(label="Wellness Score"), gr.Number(label="Insomnia Score")],
    title="AI-Powered SWAP Auto-Scorer",
    description="Upload a completed SWAP assessment to instantly get your two scores."
)

iface.launch()


Run this cell, then click the public URL. Employees can upload their filled PDF and view both scores immediately.

Step 6: (Optional) Deploy on Hugging Face Spaces
- Create a Hugging Face account and a new Space using the Gradio SDK.
- Push your app.py (all code) and requirements.txt (pdfplumber, gradio, transformers, torch, sentencepiece) to the repo.
- Wait ~1 minute for automatic deployment.
Share the Space link with your team—no Colab sign-in required.

Tips for Smooth Implementation
- If JSON parsing fails, print out resp to debug the LLM output.
- Adjust prompt temperature to 0.0–0.2 for more deterministic scoring.
- Experiment with a smaller model if you hit GPU memory limits.
- To log results, integrate with Google Sheets via gspread.
