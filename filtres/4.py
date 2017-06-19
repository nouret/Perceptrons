import cv2
import numpy

img = cv2.imread("./lena.bmp", cv2.IMREAD_COLOR)
#Преобразуем изображение из цветного в серое. 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Записываем серое изображение в файл. 
cv2.imwrite("./lena_gray.bmp", img_gray)