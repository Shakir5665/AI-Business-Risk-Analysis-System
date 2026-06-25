from src.encoding.sentiment_encoder import SentimentEncoder


encoder = SentimentEncoder()

encoder.load(

    "models/encoders/sentiment_encoder.pkl"

)

print(

    encoder.encode("positive")

)

print(

    encoder.decode(0)

)