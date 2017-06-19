import cv2
import numpy

img = cv2.imread("./lena.bmp", cv2.IMREAD_COLOR)
#Преобразуем изображение из цветного в серое. 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
X, Y = img.size // img[0].size, img[0].size // 3
print(X, Y)
sepia = (20, 66, 112) #BGR
for i in range(X):
	for j in range(Y):
		#print (img_gray[i][j])
		#print (img[i][j])
		for k in range(3):
			img[i][j][k] = int(img_gray[i][j] / 255 * sepia[k])
			#img[i][j][k] = sepia[k]
		#img_gray[i][j] = (int(((img_gray[i][j] + img_gray[i][j] + img_gray[i][j]) / 3) / 255 * sepia[0]))
#Записываем серое изображение в файл. 
cv2.imwrite("./lena_sepia.bmp", img)