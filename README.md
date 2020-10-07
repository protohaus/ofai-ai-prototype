# ofai-ai-prototype
Prototype development of the AI for the Open Farming AI Project

The first model is based on this blog post: https://towardsdatascience.com/plant-ai-plant-disease-detection-using-convolutional-neural-network-9b58a96f2289

Added second model for experiments and comparison. Switch by changing mdl in SDGModelTrainer.py.

Saved model and data are version controlled by DVC.

The images in DataPreaparation\raw form an inbalanced dataset, meaning there are different amounts of images per class. The script Data_Preaparation.py creates a balanced data subset with only those classes, that contain at least (min_images+1).