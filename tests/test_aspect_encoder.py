from src.encoding.aspect_encoder import AspectEncoder

encoder = AspectEncoder()

print("\nClasses")

print(encoder.classes)

print("\nEncoding")

print(

    encoder.encode(

        ["quality"]

    )

)

print(

    encoder.encode(

        ["quality", "trust"]

    )

)

print(

    encoder.encode(

        ["delivery"]

    )

)

print(

    encoder.encode(

        ["quality", "delivery"]

    )

)

print("\nDecoding")

print(

    encoder.decode(

        [1,0,0]

    )

)

print(

    encoder.decode(

        [1,1,0]

    )

)

print(

    encoder.decode(

        [0,1,1]

    )

)

print(

    encoder.decode(

        [1,0,1]

    )

)

encoder.save(

    "models/encoders/aspect_encoder.pkl"

)

print("\nSaved Successfully")