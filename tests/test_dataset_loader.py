from src.dataset.dataset_loader import DatasetLoader


loader = DatasetLoader()

train_df = loader.load_train()

validation_df = loader.load_validation()

test_df = loader.load_test()


print()

print("Training Samples")

print(len(train_df))

print()

print("Validation Samples")

print(len(validation_df))

print()

print("Testing Samples")

print(len(test_df))

print()

print("Columns")

print(train_df.columns)

print()

print("First Review")

print(train_df.iloc[0])