# head_fusion

1.run handleIEMOCAP to rename files in IEMOCAP corpus.

2.move the renamed files to a dictionary (e.g. '/data/*.wav').

3.run train.py to train a model and evaluate.

--------------------------------------------------
updated by 2020.5.4

To conduct noise experiments, please use train_noise.py(for influence of noise intensity and offset) and train_additional_noise.py(for data augmentation)

The original experiment results have been upload in the folder 'noise_experiment'
