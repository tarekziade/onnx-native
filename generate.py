from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
encoded = tokenizer("I think this is wonderful", return_tensors="pt")
print(encoded)
