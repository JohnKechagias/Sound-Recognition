import pathlib

workspace_dir = pathlib.Path(__file__).parent.parent.resolve()
samples_path = workspace_dir / "samples"
sample_path = samples_path / "numbers.wav"
extracted_samples_path = workspace_dir / "extracted_samples"
dataset_path = workspace_dir / "dataset"
