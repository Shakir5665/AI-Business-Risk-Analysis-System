from src.tokenization.tokenizer import ReviewTokenizer

tokenizer = ReviewTokenizer()

review = "Product eka original nemei but sound eka hodai."

result = tokenizer.tokenize(review)

print("\nInput IDs")

print(result["input_ids"])

print()

print("Attention Mask")

print(result["attention_mask"])

print()

print("Decoded Text")

print(

    tokenizer.decode(

        result["input_ids"][0]

    )

)

tokenizer.save(

    "models/tokenizer"

)

print("\nTokenizer Saved Successfully")