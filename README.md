# ofai-ai-prototype
Prototype development of the AI for the Open Farming AI Project

The first model is based on this blog post: https://towardsdatascience.com/plant-ai-plant-disease-detection-using-convolutional-neural-network-9b58a96f2289

To get started, install dvc, configure remote storage:

dvc remote add -d storage s3://my-bucket/dvc-storage

then:

dvc checkout

A virtual environment is recommended:

 py -m venv myenv c:\path\to\myenv
 
 pip install -r src/requirements.txt
 
 to install all required packages.

Added second model for experiments and comparison. Switch by changing mdl in SDGModelTrainer.py.

Model names can be found in ModelDefinitions.py.

Saved model and data are version controlled by DVC.

Training happens in SDGModelTrainer.py. To train on different data sets, change the path of data_dir. Increase the number of EPOCHS to (hopefully) increase accuracy. The training algorhythm uses checkpoint callbacks, so training can be interrupted and restarted. This is controlled by the model name. Models are stored in /AIPrototypeTrain/ckpt/{model_name}.

Pretrained models can be found in Trained_Models/{model_name}. They can be loaded with the keras.models load_model command.

SDGModelTrainer supports the use of Tensorboard. To use enter 'tensorboard --logdir=.\AIPrototypeTrain\logs' in a console. If tensorboard is not found try 'py -m tensorboard.main --logdir=.\AIPrototypeTrain\logs'. In a browser, open localhost:6006 to use Tensorboard. Wait one 'UpdateFrequency' (ie one batch, one epoch) before scalar data is availabe.

The images in DataPreparation\raw form an inbalanced dataset, meaning there are different amounts of images per class. The script Data_Preparation.py creates a balanced data subset with only those classes, that contain at least (min_images+1).


