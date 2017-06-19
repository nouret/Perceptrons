import cv2
import numpy

#Считывание изображения из файла. 
img = cv2.imread("./lena.bmp", cv2.IMREAD_COLOR)#и 
#инициализация им матрицы img.
 
cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img)
cv2.waitKey(0)
