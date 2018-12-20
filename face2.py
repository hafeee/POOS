import cv2
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import glob
from resizeimage import resizeimage


def spremiSliku(slika, filename,path):
    cv2.imwrite(os.path.join(path , filename[8:]),slika)
    print("Uspjesno spasena slika u: " + path + " " + filename[8:])

def filter(image, k,filename):
    blur = cv2.blur(image,(k,k))
    path = 'zamagljeneSlike'
    spremiSliku(blur,filename,path)
    return blur

def filter2(image, k,filename):
    dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    path ='otklanjanjeSumaSlike'
    spremiSliku(dst,filename,path)
    return dst

def svjetlina(image, brightness = 0):
    beta =50
    res = cv2.add(image, beta) 
    return res

def kontrast(image, contrast=0):
    buf=[]
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f) 

        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return buf  

def histogram(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def prikazi(img, filt,orginal):
    plt.subplot(131),plt.imshow(img),plt.title('prvi')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(filt),plt.title('drugi')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(orginal),plt.title('orginal')
    plt.xticks([]), plt.yticks([]) 
    plt.show()

def podijeliSlike(path):
    brojac = 0
    for filename in glob.glob(path):
        image = cv2.imread(filename)
        if brojac % 5 == 0:
            spremiSliku(image, filename, "test")
        else:
            spremiSliku(image, filename, "train")
        brojac = brojac + 1

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]




def prepare_training_data(data_folder_path): 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
    
    #our subject directories start with letter 's' so
    #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue
    
    #------STEP-2--------
    #extract label number of subject from dir_name
    #format of dir name = slabel
    #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
        
        #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
        
        #build image path
        #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
        
        #read image
            image = cv2.imread(image_path)
        
        #display an image window to show the image 
            #cv2.imshow("Training on image...", image)
            #cv2.waitKey(100)

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
            #DOVLE

            #cv2.imshow("Training on image with descriptor...", image)
            #cv2.waitKey(100)


        #detect face
            face, rect = detect_face(image)

            
        #------STEP-4--------
        #for the purpose of this tutorial
        #we will ignore faces that are not detected
            if face is not None:
        #add face to list of faces
                faces.append(face)
        #add label for this face
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
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)




TP = 0 #true positive
FP = 0
TN = 0
FN = 0


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img, lice):
    print("LICE: ", lice)
    global TP #true positive
    global FP
    global TN
    global FN
    subjects = ["", "face", "noFace"]
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    try:
        label= face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        if label[0] == 1 and lice == True:
            TP = TP + 1
        elif label[0] == 1 and lice == False:
            FP = FP + 1
        label_text = subjects[label[0]]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
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



print("Preparing data...")
faces, labels = prepare_training_data("train")
print("Data prepared")
 
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.createLBPHFaceRecognizer()

face_recognizer.train(faces, np.array(labels))

print("Predicting images...")
for filename in glob.glob('test/*'):
 
    #load test images
    test_img = cv2.imread(filename)
    lice = False
    if filename[5]=='l':
        lice = True 

    print(filename)
    #perform a prediction
    predicted_img = predict(test_img, lice)

    cv2.imshow("slika", predicted_img)
    cv2.waitKey(100)
    print("TP:",TP,TN,FP,FN)

print("Prediction complete")
cv2.waitKey(0)
cv2.destroyAllWindows()

#Performanse modela
ukupno = TP + FP + TN + FN
acc = float(TP + TN) / ukupno
sen = TP / float(TP + FN)
spe = TN / float(FP + TN)
print("ACCURACY: ", acc)
print("SENSITIVITY: ", sen)
print("SPECIFICITY: ", spe)
'''
file = open("anotacije.txt","w") 
for filename in glob.glob('dataset/*'):
    
    im = Image.open(filename)
    imagePath = filename
    cascPath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(imagePath)

    width,height=im.size
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    img = np.zeros([height,width,3],dtype=np.uint8)
    img.fill(0)

    imResize = im
    josNovijaSlika = image.copy()
    novaSlika = image.copy()
    for (x, y, w, h) in faces:
        file.write(filename + "  " + str(x) + "  " +  str(y) + "  " +  str(w) + "  " +  str(h)+ "\n") #Anotacija
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)      
        cv2.rectangle(novaSlika, (x, y), (x+w, y+h), (255, 255, 255), -1)  
        maxi = w
        if h > maxi:
            maxi = h
        area = (x, y, maxi+x, y+maxi)
        cropped_img = im.crop(area)
        imResize = cropped_img.resize((128,128), Image.ANTIALIAS)
        

        pil_image = imResize.convert('RGB') 
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        kontrast_nad_maskom=svjetlina(open_cv_image,50)

        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(kontrast_nad_maskom,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(kontrast_nad_maskom, kp)
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(kontrast_nad_maskom, kp, None, color=(0,255,0), flags=0)
        plt.imshow(img2), plt.show()

'''

'''
masked_data = cv2.bitwise_and(image, img)

#spremiSliku(masked_data, filename,'maskiraneSlike')

#slika = filter(image, 5, filename)
#filter2(slika, 5, filename)

#spremiSliku(histogram(kontrast(svjetlina(image,50),20)), filename,'editovaneSlike')
'''


#file.close()

#podijeliSlike("dataset/*")

