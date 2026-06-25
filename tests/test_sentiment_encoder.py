from src.encoding.sentiment_encoder import SentimentEncoder


encoder = SentimentEncoder()

print(

    encoder.classes

)

print(

    encoder.encode("positive")

)

print(

    encoder.encode("negative")

)

print(

    encoder.encode("neutral")

)

print(

    encoder.decode(2)

)

encoder.save(

    "models/encoders/sentiment_encoder.pkl"

)

print("Saved successfully.")