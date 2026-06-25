from src.pipeline.pipeline_components import PipelineComponents

components = PipelineComponents()

print()

print("Tokenizer")

print(type(components.tokenizer))

print()

print("Sentiment Encoder")

print(type(components.sentiment_encoder))

print()

print("Aspect Encoder")

print(type(components.aspect_encoder))