# deepfake-detection

Identify deepfake videos using SVM Classifier.

A set of real and deepfake videos are provided. We trained a ML model using SVM to learn from the given real-fake classification and use that ML model to classify given a new video, if it is real or fake. 

svm-classification.ipynb is a notebook which includes steps to perform extracting the features from the videos. metadata.json file located in the training video dataset provided information if the video provided is real or fake. It also included providing mapping of the original real video for a given fake video. We performed feature engineering steps using pandas library to extract the real and fake videos and write them into two different directories. Although we created 100s of real and fake videos, we picked a total of 60 real videos and 60 fake videos, as we were limited by compute resources of our laptops. We split the dataset to 70% for training and 30% for testing. open-cv library helped in capturing frames from the video. We captured seven frames for each video. We used imageio library to read these images and flatten them to numpy.ndarray, which can act as features. Scikit-learn offers SVM library and we leveraged C-Support Vector Classification algorithm to train the model. Once the model is built, which takes many hours to train, we saved the model in pickle format for later use. The file turned out to be over 24 GB file size. We used the test split of 30% for all our testing. We generated multiple metrics including confusion-matrix, ROC curve, precision, recall, f1-score, accuracy score and log loss score. Although the ML model did not perform very well with a low accuracy score, this is a humble start given our limited compute resource and training time. 

 


/dataviz.ipynb : Jupyter notebook containing code for the data visualizations <br>
/haarcascade_frontalface_default.xml : xml file containing the haar cascade face detection model

