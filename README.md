# Deepfake Detection: Important Directories and Files
/eda/: Directory containing tools for exploratory data analysis.<br>
/eda/dataviz.ipynb: Jupyter notebook containing code for the data visualizations.<br>
/eda/haarcascade_frontalface_default.xml: XML file containing the haar cascade face detection model.

/preprocessing/: Directory containing implementations of data pre-processing techniques.<br>
/preprocessing/mask_rcnn/mask_rcnn.py: Contains inference code for Mask R-CNN loaded with pre-trained weights from training on COCO dataset.<br>
/preprocessing/mask_rcnn/utils.py: Contains helper functions for Mask R-CNN inference.

/models/: Directory containing model implementations<br>

/models/svm: Directory containing implementation of SVM model for deepfake classification.<br>
/models/svm/svm-classification.ipynb: Jupyter notebook that includes video feature extraction steps and SVM model implementation for deepfake classification. Additionally includes code for calculating model performance on metrics discussed in the results section of the midterm checkpoint and visualizations based on these metrics.

/models/vit/: Directory containing EfficientNet-ViT model implementation and environment creation file.<br>
/models/vit/efficient-vit/: Directory containing implementation of EfficientNet-ViT model for deepfake classification.<br>
/models/vit/environment.yml: Environment file to create conda environment for training and testing.<br>
/models/vit/efficient-vit/efficient_vit.py: Contains implementation of EfficientNet-ViT model using PyTorch's out-of-the-box implementation of EfficientNet and a transformer implementation using PyTorch and NumPy.<br>
/models/vit/efficient-vit/train.py: Contains training procedure for EfficientNet-ViT model.<br>
/models/vit/efficient-vit/test.py: Contains inference code for trained EfficientNet-ViT model. Loads trained model from checkpoint and evaluates on metrics discussed in the results section of the midterm checkpoint. Generates visualizations based on model performance on these metrics.<br>
/models/vit/efficient-vit/utils.py: Contains helper functions for training and inference processes.<br>
/models/vit/efficient-vit/configs/architecture.yml: Config file for adjusting model hyperparameters used during training.

/models/ensemble/DenseNet121.pynb: Contains training pipeline for DenseNet121 Model for ensemble model 
/models/ensemble/InceptionResNet.pynb: Contains training pipeline for InceptionResNet model for ensemble model
/models/ensemble/Ensemble.pynb: Contains both models loaded as well as the LightGBM to create the meta learner
