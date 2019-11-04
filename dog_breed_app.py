import numpy as np
from PIL import Image
import base64
import io
import cv2                
import matplotlib.pyplot as plt 
from extract_bottleneck_features import extract_Resnet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image               
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint 
from flask import request, jsonify, Flask


app = Flask(__name__)

dog_names = ['ages/train/001.Affenpinscher',
 'ages/train/002.Afghan_hound',
 'ages/train/003.Airedale_terrier',
 'ages/train/004.Akita',
 'ages/train/005.Alaskan_malamute',
 'ages/train/006.American_eskimo_dog',
 'ages/train/007.American_foxhound',
 'ages/train/008.American_staffordshire_terrier',
 'ages/train/009.American_water_spaniel',
 'ages/train/010.Anatolian_shepherd_dog',
 'ages/train/011.Australian_cattle_dog',
 'ages/train/012.Australian_shepherd',
 'ages/train/013.Australian_terrier',
 'ages/train/014.Basenji',
 'ages/train/015.Basset_hound',
 'ages/train/016.Beagle',
 'ages/train/017.Bearded_collie',
 'ages/train/018.Beauceron',
 'ages/train/019.Bedlington_terrier',
 'ages/train/020.Belgian_malinois',
 'ages/train/021.Belgian_sheepdog',
 'ages/train/022.Belgian_tervuren',
 'ages/train/023.Bernese_mountain_dog',
 'ages/train/024.Bichon_frise',
 'ages/train/025.Black_and_tan_coonhound',
 'ages/train/026.Black_russian_terrier',
 'ages/train/027.Bloodhound',
 'ages/train/028.Bluetick_coonhound',
 'ages/train/029.Border_collie',
 'ages/train/030.Border_terrier',
 'ages/train/031.Borzoi',
 'ages/train/032.Boston_terrier',
 'ages/train/033.Bouvier_des_flandres',
 'ages/train/034.Boxer',
 'ages/train/035.Boykin_spaniel',
 'ages/train/036.Briard',
 'ages/train/037.Brittany',
 'ages/train/038.Brussels_griffon',
 'ages/train/039.Bull_terrier',
 'ages/train/040.Bulldog',
 'ages/train/041.Bullmastiff',
 'ages/train/042.Cairn_terrier',
 'ages/train/043.Canaan_dog',
 'ages/train/044.Cane_corso',
 'ages/train/045.Cardigan_welsh_corgi',
 'ages/train/046.Cavalier_king_charles_spaniel',
 'ages/train/047.Chesapeake_bay_retriever',
 'ages/train/048.Chihuahua',
 'ages/train/049.Chinese_crested',
 'ages/train/050.Chinese_shar-pei',
 'ages/train/051.Chow_chow',
 'ages/train/052.Clumber_spaniel',
 'ages/train/053.Cocker_spaniel',
 'ages/train/054.Collie',
 'ages/train/055.Curly-coated_retriever',
 'ages/train/056.Dachshund',
 'ages/train/057.Dalmatian',
 'ages/train/058.Dandie_dinmont_terrier',
 'ages/train/059.Doberman_pinscher',
 'ages/train/060.Dogue_de_bordeaux',
 'ages/train/061.English_cocker_spaniel',
 'ages/train/062.English_setter',
 'ages/train/063.English_springer_spaniel',
 'ages/train/064.English_toy_spaniel',
 'ages/train/065.Entlebucher_mountain_dog',
 'ages/train/066.Field_spaniel',
 'ages/train/067.Finnish_spitz',
 'ages/train/068.Flat-coated_retriever',
 'ages/train/069.French_bulldog',
 'ages/train/070.German_pinscher',
 'ages/train/071.German_shepherd_dog',
 'ages/train/072.German_shorthaired_pointer',
 'ages/train/073.German_wirehaired_pointer',
 'ages/train/074.Giant_schnauzer',
 'ages/train/075.Glen_of_imaal_terrier',
 'ages/train/076.Golden_retriever',
 'ages/train/077.Gordon_setter',
 'ages/train/078.Great_dane',
 'ages/train/079.Great_pyrenees',
 'ages/train/080.Greater_swiss_mountain_dog',
 'ages/train/081.Greyhound',
 'ages/train/082.Havanese',
 'ages/train/083.Ibizan_hound',
 'ages/train/084.Icelandic_sheepdog',
 'ages/train/085.Irish_red_and_white_setter',
 'ages/train/086.Irish_setter',
 'ages/train/087.Irish_terrier',
 'ages/train/088.Irish_water_spaniel',
 'ages/train/089.Irish_wolfhound',
 'ages/train/090.Italian_greyhound',
 'ages/train/091.Japanese_chin',
 'ages/train/092.Keeshond',
 'ages/train/093.Kerry_blue_terrier',
 'ages/train/094.Komondor',
 'ages/train/095.Kuvasz',
 'ages/train/096.Labrador_retriever',
 'ages/train/097.Lakeland_terrier',
 'ages/train/098.Leonberger',
 'ages/train/099.Lhasa_apso',
 'ages/train/100.Lowchen',
 'ages/train/101.Maltese',
 'ages/train/102.Manchester_terrier',
 'ages/train/103.Mastiff',
 'ages/train/104.Miniature_schnauzer',
 'ages/train/105.Neapolitan_mastiff',
 'ages/train/106.Newfoundland',
 'ages/train/107.Norfolk_terrier',
 'ages/train/108.Norwegian_buhund',
 'ages/train/109.Norwegian_elkhound',
 'ages/train/110.Norwegian_lundehund',
 'ages/train/111.Norwich_terrier',
 'ages/train/112.Nova_scotia_duck_tolling_retriever',
 'ages/train/113.Old_english_sheepdog',
 'ages/train/114.Otterhound',
 'ages/train/115.Papillon',
 'ages/train/116.Parson_russell_terrier',
 'ages/train/117.Pekingese',
 'ages/train/118.Pembroke_welsh_corgi',
 'ages/train/119.Petit_basset_griffon_vendeen',
 'ages/train/120.Pharaoh_hound',
 'ages/train/121.Plott',
 'ages/train/122.Pointer',
 'ages/train/123.Pomeranian',
 'ages/train/124.Poodle',
 'ages/train/125.Portuguese_water_dog',
 'ages/train/126.Saint_bernard',
 'ages/train/127.Silky_terrier',
 'ages/train/128.Smooth_fox_terrier',
 'ages/train/129.Tibetan_mastiff',
 'ages/train/130.Welsh_springer_spaniel',
 'ages/train/131.Wirehaired_pointing_griffon',
 'ages/train/132.Xoloitzcuintli',
 'ages/train/133.Yorkshire_terrier']
#Pre-process the data
def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    x = image.img_to_array(img)
    
    return  np.expand_dims(x, axis = 0)

print(" * Loading Keras model...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Detect if human
def face_detector(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
def ResNet50_predict_labels(img):
    # returns prediction vector for image located at img_path
    img = preprocess_input(img)
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img):
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 


# get model
bottleneck_features = np.load('DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']


Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

Resnet50_model.load_weights('weights.best.Resnet50.hdf5')


def Resnet50_predict_breed(tensor):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(tensor)
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

print(" * model loaded!")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    imagem = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(imagem, target_size=(224,224))
        
    if dog_detector(processed_image):
        response = {
        'prediction':{
            'answer':"It's a dog! It seems to be a {}".format(Resnet50_predict_breed(processed_image).split(".")[1])
        }
        }
        return response 
        
    if face_detector(imagem):
        response = {
        'prediction':{
            'answer':"It's a human!" +\
            "If it was a dog it would be a {}".format(Resnet50_predict_breed(processed_image).split(".")[1])
        }
        }
        return response 
        
    else:
        response = {
        'prediction':{
            'answer':"It does not seems to be a dog or a human."
        }
        }
        return response 
          
    return jsonify(response)