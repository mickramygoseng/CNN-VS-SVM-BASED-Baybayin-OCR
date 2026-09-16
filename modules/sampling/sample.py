from sampling import Sample

svm_dataset = "data/processed_dataset/svm_dataset.pkl"
cnn_dataset = "data/processed_dataset/cnn_dataset.pkl"

svm_target_dir = "data/sampled_dataset/svm"
cnn_target_dir = "data/sampled_dataset/cnn"

folds = 5
repeat = 10
seed = 42

# For SVM
svm_sample = Sample(
    mainDataset=svm_dataset,
    targetDir=svm_target_dir
)
svm_sample.RSKF(
    folds=folds,
    repeat=repeat,
    seed=seed
)

# For CNN
cnn_sample = Sample(
    mainDataset=cnn_dataset,
    targetDir=cnn_target_dir
)
cnn_sample.RSKF(
    folds=folds,
    repeat=repeat,
    seed=seed
)