import cv2
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import glob
from resizeimage import resizeimage



def svjetlina(image, brightness = 0):
    beta =50
    res = cv2.add(image, beta) 
    return res



def prikazi_jednu(img):
    plt.imshow(img),plt.title('Slika')
    plt.xticks([]), plt.yticks([])
    plt.show()





def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    
    if (len(faces) == 0):
        return None, None

    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def resizeAndGrayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return gray
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h]

def obradiSliku(image):
    image = resizeAndGrayscale(image)
    #Metoda poboljsavanja
    image=svjetlina(image,50)

    #dodani deskriptor
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(image,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    # draw only keypoints location,not size and orientation
    image = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    
    return image

def prepare_training_data(data_folder_path): 
    #Dobaviti foldere
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []
    
    for dir_name in dirs:
    
        if not dir_name.startswith("s"):
            continue
    
        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        
        for image_name in subject_images_names:
        
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
        
            image = cv2.imread(image_path)
            
            image = obradiSliku(image)

            #cv2.imshow("Training on image with descriptor...", image)
            #cv2.waitKey(100)


            #detect face
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                faces.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                labels.append(label)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

TP = 0 #true positive
FP = 0 #false positive
TN = 0 #true negative
FN = 0 #false negative


def predict(test_img, face_recognizer, lice):
    global TP #true positive
    global FP
    global TN
    global FN
    subjects = ["", "face", "noFace"]

    img = test_img.copy()

    #detect face from the image
    face, rect = detect_face(img)

    try:
        label= face_recognizer.predict(face)

        if label[0] == 1 and lice == True:
            TP = TP + 1
        elif label[0] == 1 and lice == False:
            FP = FP + 1
        label_text = subjects[label[0]]
        
        draw_rectangle(img, rect)

        draw_text(img, label_text, rect[0], rect[1]-5)
        
    except:

        label_text = subjects[2]
        draw_text(img, label_text, 50, 50)
        if lice == True:
            FN = FN + 1
        else:
            TN = TN + 1
        #label_text = subjects[label[0]]
    
    return img


def main(filePath):
    for filename in glob.glob(filePath):
        validacijaSlika = cv2.imread(filename)

        #Deskriptori i poboljsanja 2,3
        validacijaSlika = obradiSliku(validacijaSlika)

        face_recognizer = cv2.face.createLBPHFaceRecognizer()
        face_recognizer.load("trainingdata.yml")

        validacijaSlika = predict(validacijaSlika, face_recognizer, False)

        cv2.imshow("slika", validacijaSlika)
        cv2.waitKey(1000)


def train():
    print("Preparing data...")
    faces, labels = prepare_training_data("train")
    print("Data prepared")

    #create our LBPH face recognizer 
    face_recognizer = cv2.face.createLBPHFaceRecognizer()

    face_recognizer.train(faces, np.array(labels))
    #export treniranog modela
    face_recognizer.save("trainingdata.yml")
    testing(face_recognizer)

def testing(face_recognizer):
    print("Predicting images...")
    for filename in glob.glob('test/*'):
    
        #load test images
        test_img = cv2.imread(filename)

        #test_img = obradiSliku(test_img)

        lice = False
        if filename[5]=='l':
            lice = True 

        print(filename)
        #perform a prediction
        predicted_img = predict(test_img, face_recognizer, lice)

        #cv2.imshow("slika", predicted_img)
        #cv2.waitKey(100)
        print("TP:",TP,TN,FP,FN)

    print("Prediction complete")
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Performanse modela
    ukupno = TP + FP + TN + FN
    acc = float(TP + TN) / ukupno
    sen = TP / float(TP + FN)
    spe = TN / float(FP + TN)
    print("ACCURACY: ", acc)
    print("SENSITIVITY: ", sen)
    print("SPECIFICITY: ", spe)


train()
main("validacija/*")
