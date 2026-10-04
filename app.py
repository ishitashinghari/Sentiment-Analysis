import gradio as gr
from transformers import pipeline

import os

classifier = pipeline('sentiment-analysis')
def sentiment(text):
  if not text.strip():
    return "Enter a valid text","0.00%"
  result=classifier(text)[0]
  label=result['label']
  score=result['score']*100
  return label,f"{score:.2f}%"

app=gr.Interface(
    fn=sentiment,
    inputs="text",
    outputs=["text","text"],
    title="Sentiment-Analysis"
)
port=int(os.environ.get("PORT",10000))
app.launch(server_name='0.0.0.0',server_port=port)
