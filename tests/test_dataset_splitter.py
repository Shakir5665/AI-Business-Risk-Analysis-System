from src.dataset.splitter import DatasetSplitter

splitter = DatasetSplitter()

splitter.execute(

    dataset_path="data/processed/dataset.jsonl",

    output_directory="data/processed"

)