# ofai-ai-prototype
Prototype development of the AI for the Open Farming AI Project

The first model is based on this blog post: https://towardsdatascience.com/plant-ai-plant-disease-detection-using-convolutional-neural-network-9b58a96f2289

Added second model for experiments and comparison. Switch by changing mdl in SDGModelTrainer.py.

Saved model and data are version controlled by DVC.

The images in DataPreaparation\raw form an inbalanced dataset, meaning there are different amounts of images per class. The script Data_Preparation.py creates a balanced data subset with only those classes, that contain at least (min_images+1).

SDGModelTrainer supports the use of Tensorboard. To use enter 'tensorboard --logdir=.\AIPrototypeTrain\logs' in a console. If tensorboard is not found try 'py -m tensorboard.main --logdir=.\AIPrototypeTrain\logs'. In a browser, open localhost:6006 to use Tensorboard. Wait one 'UpdateFrequency' (ie one batch, one epoch) before scalar data is availabe.
