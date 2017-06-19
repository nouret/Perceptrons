import cv2
import numpy

img = numpy.ndarray((400, 500, 3), numpy.uint8) #Создание изображения 
#имеющее размер 500 в ширину на 400 в высоту. 

#Для удобства, аргументы вызываемой функции: 
text = "Hello World!" #Целевая фраза. 
textOrg = (100, img.shape[1] // 2) #Местоположение
fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX #Фонт 
fontScale = 2 #Его масштаб, размер 
color = (200, 100, 50) #Цвет текста. 
thickness = 3 #толщина линий фонта. 

#Далее выполняется отрисовка фразы из переменной 
#text в изображение img в положение textOrg. 
#Фонт/гарнитура задается следующими переменными: 
#fontFace, fontScale, thickness. 
cv2.putText(img, text, textOrg, fontFace, fontScale, color, thickness, 8)
cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img)
cv2.waitKey(0)
