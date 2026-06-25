from src.tokenization.tokenizer import ReviewTokenizer

tokenizer = ReviewTokenizer()

review = "Product eka original nemei but sound eka hodai."

result = tokenizer.tokenize(review)

print("\nInput IDs Shape")
print(result["input_ids"].shape)

print("\nAttention Mask Shape")
print(result["attention_mask"].shape)

print("\nDecoded Text")
print(
    tokenizer.decode(
        result["input_ids"]
    )
)

tokenizer.save(
    "models/tokenizer"
)

print("\nTokenizer Saved Successfully")