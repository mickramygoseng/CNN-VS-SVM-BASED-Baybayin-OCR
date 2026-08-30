from sampling import Sample

dataset = "data/processed_dataset/dataset.pkl"
target_dir = "data/sampled_dataset"
folds = 5
repeat = 10
seed = 42

sample = Sample(
    mainDataset=dataset,
    targetDir=target_dir
)

sample.RSKF(
    folds=folds,
    repeat=repeat,
    seed=seed
)