import cv2 # se usa para detectar las manos
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier 
import numpy as np # Se usa para calcular los tamaños de la img 
import math # Se usa para calcular los tamaños de la img 
import pyttsx3 # Se usa para que de VOZ

cap = cv2.VideoCapture(0) # capturar el video
detector = HandDetector(maxHands=1) # Detector de la mano
ClasificadorBD = Classifier("model/keras_model.h5", "Model/labels.txt") # clasificador aqui es donde esta entrenada

Hablar = pyttsx3.init() # ejecuta el hablador 

# Tasa
Tasa = Hablar.getProperty('rate')   # obtener detalles de la tasa de habla actual
Hablar.setProperty('rate', 125)     # configurar una nueva tarifa de voz

# Volumen
volumen = Hablar.getProperty('volume')   #familiarizarse con el nivel de volumen actual (min=0 y max=1)
Hablar.setProperty('volume',100)    # configurar el nivel de volumen entre 0 y 1

# Vos
Vos = Hablar.getProperty('voices')       #obteniendo detalles de la voz actual
Hablar.setProperty('voice', Vos[0].id)   #cambiando índice, cambia voces. 0 para hombre y 1 para mujer

BordeImg = 20 # Borde extra de la imagen a reconocer o recortar
ImgTamano = 300 # Tamaño de la imagen

#Carpeta = "imagenes/O" # Carpeta donde se guardara la img
#Contador = 0 # contador inicio para guardar la img y llevar una cuenta

labels = ["A","B","BORRAR","C","D","E","F","HOLACOMOESTAS","I","L","M","O","U","V"] #Aqui va letras
letra_anterior = ''
Palabra = ""
show = " "
while True: # El programa se queda en un ciclo infinito para que nunca pare de detectar la mano
    success, img = cap.read()
    img = img.copy()
    hands, img = detector.findHands(img)

    if hands: # Si es que detecta la mano continua y se le agregan mas parametros para tener una mejor clasificacion de la mano
        hand = hands[0] # Manos a detectar
        x, y, w, h = hand['bbox'] # Se usa para darle un tamaño al delimitador de reconocimiento de imagen y que solo detecte la mano en vez de toda la pantalla 

        VentanaConFondoBlanco = np.ones((ImgTamano, ImgTamano, 3), np.uint8) * 255 # Tamaño ideal que deveria tener para guardar la img y tener un mejor reconocimiento
        VentanaRecortada = img[y - BordeImg:y + h + BordeImg, x - BordeImg:x + w + BordeImg] # Tamaño para reconocer mejor la img

        VentanaRecortadaShape = VentanaRecortada.shape # se obtiene la img y se recorta la anchura, altura 

        aspectRatio = h / w # Se calcula el tamaño de la mano

        if aspectRatio > 1: # en caso de que el tamaño sea el adeacuado para verificar la mano se procede a indetificar la mano 
            k = ImgTamano / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(VentanaRecortada, (wCal, ImgTamano))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((ImgTamano - wCal) / 2)
            VentanaConFondoBlanco[:, wGap:wCal + wGap] = imgResize
            prediction, index = ClasificadorBD.getPrediction(VentanaConFondoBlanco, draw=False) # Predice la forma de la mano y verifica en el modelo y si la letra es reconocida
            print(prediction, index) # muenstra la predicion de la coordenadas de la letra identificada

        else: # en caso de que la mano no sea el tamaño adecuado se empieza a calcular el tamaños con nuevos datos
            k = ImgTamano / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(VentanaRecortada, (ImgTamano, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((ImgTamano - hCal) / 2)
            VentanaConFondoBlanco[hGap:hCal + hGap, :] = imgResize
            prediction, index = ClasificadorBD.getPrediction(VentanaConFondoBlanco, draw=False) # Predice la forma de la mano y verifica en el modelo y si la letra es reconocida

        if index == 2: # Borra la palabra luego habla
            Palabra = Palabra[:-1] # borra la ultima palabra
            Hablar.say(Palabra)
            Hablar.runAndWait()

        #if index == : # se usa para borrar todas las letras Deberas poner el numero acordado de identificacion de la señar para borrar
        #    Paalabra = ""
        
        #if index == : # se usa para decir la palabra completa Deberas poner el numero acordado de identificacion de la señar para borrar
        #    Hablar.say(Palabra)
        #    Hablar.runAndWait()
            
        lim = len(Palabra) # Se guarda la cantidad de palabras
        letra_actual = labels[index] # letra actual para luego verifical
        if letra_actual != letra_anterior: # se  verifica que no sea igual a la letra anterior
            letra_anterior = letra_actual # se guarda la letra actual para que no haya repeticion de letra
            Hablar.say(labels[index]) # se verifica que letra es usando la funcion QueLetraEs y despues de que se verifique se guarda
            Hablar.runAndWait() # menciona la letra guardada
            if index != 2: # se verifica que no se guarde la palabra borrar
                if lim < 13: # se verifica la palabra no tenga mas de 13 caracteres
                    Palabra += labels[index] # aqui se va guardando cada letra identificada para cuando se haga la seña de hablar diga la palabra completa guardad aqui
        
        cv2.rectangle(img, (x - BordeImg, y - BordeImg-50),(x - BordeImg+90, y - BordeImg-50+50), (0, 12, 255), cv2.FILLED) # no identifica solamente le da un cuadro a la letra para mayor muestra
        cv2.putText(img, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2) # tamaño y color del texto del cuadro y identificador
        cv2.rectangle(img, (x-BordeImg, y-BordeImg),(x + w+BordeImg, y + h+BordeImg), (0, 12, 255), 4) # tamaño y color del texto del cuadro y identificador
        cv2.rectangle(img, (0,370), (500, 420), (0,0,0), -1)
        cv2.putText(img, Palabra, (0,420), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    cv2.imshow("Reconocimiento Principal", img) # ventana del reconocimiento principal de la camara
    cv2.waitKey(1)