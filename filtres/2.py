import cv2
import numpy

img = numpy.ndarray((400, 500, 3), numpy.uint8)
text = "Hello World!"
textOrg = (100, img.shape[1] // 2)
fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fontScale = 2
color = (200, 100, 50)
thickness = 3

cv2.putText(img, text, textOrg, fontFace, fontScale, color, thickness, 8)

#Запись изображения img в файл ./output.bmp:
cv2.imwrite("./output.bmp", img)
