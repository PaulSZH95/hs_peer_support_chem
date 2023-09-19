# QLORA Finetuning of LLAMA-7b

This README provides a simple demonstration and a quick summary tutorial on how to finetune the LLAMA-7b model using QLORA (Quantization for Language Representation) with the help of the TRL (Transformer Reinforcement Learning) library. We will also leverage quantization provided by bitsandbytes, which is integrated into the Hugging Face library.

## Readings

Before you begin, make sure you have the following prerequisites installed:

- TRL library: You can find the TRL library [here](https://github.com/huggingface/trl).
- Hugging face 4bit lesson [hugging_face](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- PEFT: Parameter-Efficient Fine-Tuning [peft](https://huggingface.co/blog/peft)
- DeepLearningAI: Building with Instruction-Tuned LLMs: A Step-by-Step Guide [deepAI](https://www.youtube.com/watch?v=eTieetk2dSw)

## Dataset

The model will be finetuned on 2 chemistry datasets available on Hugging Face. The process of combining the datasets can be found in the notebook provided. Alternatively, if you're not interested in the dataset generation, you can directly download the combined dataset from my [repository](https://huggingface.co/datasets/supramantest/hs_peer_support_chem).

## DATASET ONLY

The model will be finetuned on 2 chemistry datasets available on Hugging Face. The process of combining the datasets can be found in the notebook provided. Alternatively, if you're not interested in the dataset generation, you can directly download the combined dataset from my [repository](https://huggingface.co/datasets/supramantest/hs_peer_support_chem).

## INFERENCE ONLY

The finetuned model has been uploaded to Hugging Face at [hs_peer_support_chem](https://huggingface.co/supramantest/hs_peer_support_chem). Please note that you would require GPU for both inferencing and training as cpu version is too slow.You can use the model for inference with the following code:

```python
# Inference can be run separately from the training process
from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer

lora_config = LoraConfig.from_pretrained("supramantest/hs_peer_support_chem")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained("supramantest/hs_peer_support_chem")
model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map={"":0})
model = get_peft_model(model, lora_config)

from IPython.display import display, Markdown

def make_inference(prompt, context = None):
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
  outputs = model.generate(**inputs, max_new_tokens=100)
  display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))
```
