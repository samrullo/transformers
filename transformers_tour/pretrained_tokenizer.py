from transformers import AutoTokenizer

# We will use DistilBERT which is smaller version of BERT to classify emotion text
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"{model_name} pretrained tokenizer vocab size : {tokenizer.vocab_size}")
print(f"special token map : {tokenizer.special_tokens_map}")
print(f"model's maximum context size : {tokenizer.model_max_length}")

# let's test the tokenizer on a simple text
encoded_str=tokenizer.encode("This is a complicatedtest")
print(encoded_str)

for token in encoded_str:
    print(token,tokenizer.decode([token]))