---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
language:
- en
---
# RevOpsLM

🤗 **[View Model on Hugging Face](https://huggingface.co/Builder123/tinyllama-revops-finetuned)**

A language model trained on Salesforce Agentforce, NetSuite AI, and SaaS Revenue Recognition (ASC 606) concepts using a LoRA fine-tuned adapter for TinyLlama-1.1B-Chat.

## Model Description

This is a **proof-of-concept project** demonstrating LoRA fine-tuning techniques applied to a language model. The adapter was trained on 50 curated examples covering:

- **Salesforce Agentforce** (20 examples): Agent types, RAG, topics, guardrails, triggers, and analytics
- **NetSuite AI Features** (15 examples): Text Enhancer, Analytics Warehouse, Smart Alerts, and automation capabilities
- **SaaS Revenue Recognition** (15 examples): ASC 606 compliance, performance obligations, deferred revenue, and contract accounting

**Important**: This is a learning exercise with limited training data. The model demonstrates fine-tuning methodology and is not meant for production use.

## Training Details

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Examples**: 50
- **Epochs**: 3
- **Hardware**: Google Colab T4 GPU
- **Training Time**: ~8 minutes
- **LoRA Parameters**: r=8, alpha=16, dropout=0.05

## How to Use

To use this model, you'll need to load both the base model and this LoRA adapter:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_model = "Builder123/tinyllama-revops-finetuned"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)
# Generate response
prompt = "User: What is ASC 606?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

- Trained on limited data - responses may be inaccurate or incomplete
- Not suitable for production use without additional training data
- May hallucinate or provide outdated information
- Should be verified against official documentation

## Intended Use

This model is intended for:
- Educational purposes and learning about fine-tuning techniques
- Demonstrating LoRA methodology
- Portfolio/project showcase

## License

This model follows the license of the base TinyLlama model (Apache 2.0).

## Author

Created by Vladimir Parfenov.

---

