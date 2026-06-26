from src.tokenization.tokenizer import ReviewTokenizer

tokenizer = ReviewTokenizer()

review = "Product eka original nemei but sound eka hodai."

result = tokenizer.tokenize(review)

print(type(result["input_ids"]))
print(result["input_ids"])
print(result["input_ids"].shape)

print(type(result["attention_mask"]))
print(result["attention_mask"].shape)

print(tokenizer.decode(result["input_ids"]))