import cv2 # se usa para detectar las manos 
from cvzone.HandTrackingModule import HandDetector 
import numpy as np # Se usa para calcular los tamaños de la img 
import math # Se usa para calcular los tamaños de la img 
import time # Se usa para guardar la hora en que se guardo la img

cap = cv2.VideoCapture(0) # Empieza a capturar el video
detector = HandDetector(maxHands=1) # Detector de la mano

BordeImg = 20 # Borde extra de la imagen a reconocer o recortar
ImgTamano = 300 # Tamaño de la imagen

Carpeta = "imagenes/V" # Carpeta donde se guardara la img
Contador = 0 # contador inicio para guardar la img y llevar una cuenta

while True: # El programa se queda en un ciclo infinito para que nunca pare de detectar la mano
    success, img = cap.read()
    hands, img = detector.findHands(img) # detecta la mano
    if hands: # Si es que detecta la mano continua y se le agregan mas parametros para tener una mejor clasificacion de la mano
        hand = hands[0] # Manos a detectar
        x, y, w, h = hand['bbox'] # Se usa para darle un tamaño al delimitador de reconocimiento de imagen y que solo detecte la mano en vez de toda la pantalla 

        VentanaConFondoBlanco = np.ones((ImgTamano, ImgTamano, 3), np.uint8) * 255 # Tamaño ideal que deveria tener para guardar la img y tener un mejor reconocimiento
        VentanaRecortada = img[y - BordeImg:y + h + BordeImg, x - BordeImg:x + w + BordeImg] # Tamaño para reconocer mejor la img

        VentanaRecortadaShape = VentanaRecortada.shape # se obtiene la img y se recorta la anchura, altura 

        aspectRatio = h / w

        if aspectRatio > 1:
            k = ImgTamano / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(VentanaRecortada, (wCal, ImgTamano))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((ImgTamano - wCal) / 2)
            VentanaConFondoBlanco[:, wGap:wCal + wGap] = imgResize

        else:
            k = ImgTamano / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(VentanaRecortada, (ImgTamano, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((ImgTamano - hCal) / 2)
            VentanaConFondoBlanco[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Reconocimiento Cortado", VentanaRecortada) # Ventana de la img recortada para mejor clasificacion
        cv2.imshow("Reconocimiento Fondo blanco", VentanaConFondoBlanco) # Ventana con fondo blanco para mejor clasificacion y guardar

    cv2.imshow("Reconocimiento Principal", img) # ventana del reconocimiento Principal de la camara
    GuardarImg = cv2.waitKey(1)
    if GuardarImg == ord("s"): # Guarda la imagen si se preciona la tecla S
        Contador += 1 # Contador de la cantidad de img
        cv2.imwrite(f'{Carpeta}/Imagen_{time.time()}.jpg',VentanaConFondoBlanco) # Lugar donde se va a guardar la img con tiempo y la extencion jpg
        print(Contador) # Muestra en consola la cantidad de img que se va guardando