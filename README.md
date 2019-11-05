# DeepLearning_Dog_breed_project

#edit by LMPontes
### Table of Contents

Dog Breed Classification using CNN


1. [Project Overview](#overview)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

In this project, I build a convolutional neural network (CNN) that can identify whether the given image is either a dog or a human or none. 
If detected dog, then code will identify the dog breed. Otherwise, if detected human, then the code will identify the resembling dog breed.
The CNN used an pre-trained network named ResNet50, which uses imagenet images to train the ResNet50 neural network. I attached convolutional layers to the pre-trained NN in order to adapt to the dog breed subject.
The code is written in Python 3 and Keras with Tensorflow backend all presented in Jupyter Notebook.
I deployed the CNN classifier on a local html page, that you can run locally on your machine.

## Instructions <a name="instructions"></a>

The web application uses TensorFlow with GPU support on local machine. Follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.
There should be necessary to install keras libraries to run the code. To run the app it is necessary to install flask. And it is recommended that you create a virtual environment in Anaconda to install those libraries.
To run the app, open Anaconda prompt, activate the enviroment, and run (on Windows): 

```
set FLASK_APP=dog_breed_project.py
flask run --host:0.0.0.0
```

On Linux and Mac type:

```
execute FLASK_APP=dog_breed_project.py
flask run --host:0.0.0.0
```

After that, open the browser and use your local IP, type:
```
your_IP/static/predict.html
```

The application olny accepts .png image files, that can be uploaded to the html page and classified by pressing the "Classify" button.


## File Descriptions <a name="files"></a>

There is one notebook available here to showcase work related to the above questions.
The directory static contains the html configuration used for the web app.
The other files are for the pre-trained CNN and for the face and dog detector functions.  

## Results<a name="results"></a>

The dog breed classifier obtained a 80% accuracy score on the test dataset.
Have fun by finding which dog breed an human face looks like.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to 
Main Kaggle Competition link :
Dog Breed Identification (https://www.kaggle.com/c/dog-breed-identification)
