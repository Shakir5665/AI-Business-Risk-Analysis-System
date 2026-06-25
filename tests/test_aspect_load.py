from src.encoding.aspect_encoder import AspectEncoder


encoder = AspectEncoder()

encoder.load(

    "models/encoders/aspect_encoder.pkl"

)

print()

print(

    encoder.classes

)

print()

print(

    encoder.encode(

        ["trust","delivery"]

    )

)

print()

print(

    encoder.decode(

        [0,1,1]

    )

)