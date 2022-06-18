import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
image = keras.preprocessing.image

classes = { 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
            1: ('bcc' , ' basal cell carcinoma'),
            2: ('bkl', 'benign keratosis-like lesions'),
            3: ('df', 'dermatofibroma'),
            4: ('nv', ' melanocytic nevi'),
            5: ('vasc', 'vascular lesions'),
            6: ('mel', 'melanoma') }

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


def classify(image_fp):

    image = plt.imread(image_fp)
    image = crop_square(image, 28)
    plt.imshow(image)

    model = keras.models.load_model('../final_cnn.h5')
    predicted_value = model.predict(np.array([image]))
    
    out  = f"""
      akiec: {predicted_value[0][0]}
        bcc: {predicted_value[0][1]}
        bkl: {predicted_value[0][2]}
         df: {predicted_value[0][3]}
         nv: {predicted_value[0][4]}
       vasc: {predicted_value[0][5]}
        mel: {predicted_value[0][6]}

     Result: {(predicted_value[0][predicted_value.argmax()] * 100)}% {classes[predicted_value.argmax()]}
    """
    return out