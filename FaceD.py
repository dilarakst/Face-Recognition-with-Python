
import cv2
import numpy as np

#Yüzün tanımlanmasını sağlayan gönderdiğimiz cascade.
yuz_tanima = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#İçeri videoyu VideoCapture sayesinde aktarıyoruz.
video = cv2.VideoCapture("video0.mp4") 

while True:
    _,kare = video.read()
    karex = cv2.pyrUp(kare)
    #Videoyu rgbden arındırarak OpenCV'nin daha iyi algılamasını sağlıyorum.
    gri_kare = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY) 
    gri_karex = cv2.pyrUp(gri_kare)

    #Cascade tanımlama
    yuzler = yuz_tanima.detectMultiScale(gri_karex,1.1,4) 

    print(yuzler)

    #Yüzleri kare içerine alma
    for (x,y,w,h) in yuzler:
        cv2.rectangle(karex,(x,y),(x+w,y+h),(0,255,0),3)

    #Pencere oluşturup aktarım sağlar
    cv2.imshow("dilara",karex)

    #A tuşuna basınca videodan çıkar
    if cv2.waitKey(10) & 0xFF == ord("a"):
        break


video.release()
cv2.destroyAllWindows()