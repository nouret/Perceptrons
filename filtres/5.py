import cv2
import numpy

img = cv2.imread("./lena.bmp", cv2.IMREAD_COLOR)
whitematrix = numpy.copy(img) #создаём картинку нужного
#размера, правильнее было бы достать размер из img
#размеры картинки и создать соответствующую матрицу
whitematrix.fill(255) #операция int - numpy.array не
#определена, создаём отдельно белую картинку
img_neg = whitematrix - img #Вычисляем негатив.
cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img_neg)
cv2.waitKey(0)
