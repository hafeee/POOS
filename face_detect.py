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
    masked_data = cv2.bitwise_and(image, img)

    #spremiSliku(masked_data, filename,'maskiraneSlike')

    #slika = filter(image, 5, filename)
    #filter2(slika, 5, filename)
    
    #spremiSliku(histogram(kontrast(svjetlina(image,50),20)), filename,'editovaneSlike')
    '''
    slike=kontrast(image,30)
    cv2.imshow("", slike)
    cv2.waitKey(0)


file.close()

#podijeliSlike("dataset/*")

