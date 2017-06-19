import cv2
import numpy

numpy.random.seed(179)

N = 500
img = numpy.ndarray((N, N, 3), numpy.uint8)

img += 255

plus = 0
while plus < 50:
    x0 = numpy.random.randint(5, N - 5)
    y0 = numpy.random.randint(5, N - 5)
    #print(x0, y0)
    Color = [(x0, y0)]
    NotColor = []
    for i in list(range(-5, 0)) + list(range(1, 6)):
        Color += [(x0 + i, y0)]
        NotColor += [(x0 + i, y0 + 1)]
        NotColor += [(x0 + i, y0 - 1)]
    NotColor += [(x0 - 6, y0), (x0 - 6, y0 - 1), (x0 - 6, y0 + 1)]
    NotColor += [(x0 + 6, y0), (x0 + 6, y0 - 1), (x0 + 6, y0 + 1)]
    for i in list(range(-5, 0)) + list(range(1, 6)):
        Color += [(x0, y0 + i)]
        NotColor += [(x0 + 1, y0 + i)]
        NotColor += [(x0 - 1, y0 + i)]
    NotColor += [(x0, y0 - 6), (x0 - 1, y0 - 6), (x0 + 1, y0 - 6)]
    NotColor += [(x0, y0 + 6), (x0 - 1, y0 + 6), (x0 + 1, y0 + 6)]
    can = True
    for elem in set(NotColor):
        x, y = elem
        if 0 <= x < N and 0 <= y < N and not(img[x][y][0] == img[x][y][1] == img[x][y][2] == 255):
            can = False
            break
    B, G, R = numpy.random.randint(0, 100), numpy.random.randint(0, 100), numpy.random.randint(155, 255)
    if can:
        for elem in set(Color):
            x, y = elem
            img[x][y][0] = B
            img[x][y][1] = G
            img[x][y][2] = R
        plus += 1


minus = 0
while minus < 50:
    x0 = numpy.random.randint(5, N - 5)
    y0 = numpy.random.randint(5, N - 5)
    Color = [(x0, y0)]
    NotColor = []
    for i in list(range(-5, 0)) + list(range(1, 6)):
        Color += [(x0, y0 + i)]
        NotColor += [(x0 + 1, y0 + i)]
        NotColor += [(x0 - 1, y0 + i)]
    NotColor += [(x0, y0 - 6), (x0 - 1, y0 - 6), (x0 + 1, y0 - 6)]
    NotColor += [(x0, y0 + 6), (x0 - 1, y0 + 6), (x0 + 1, y0 + 6)]
    can = True
    for elem in set(NotColor):
        x, y = elem
        if 0 <= x < N and 0 <= y < N and not(img[x][y][0] == img[x][y][1] == img[x][y][2] == 255):
            can = False
            break
    B, G, R = numpy.random.randint(0, 100), numpy.random.randint(155, 255), numpy.random.randint(0, 100)
    if can:
        for elem in set(Color):
            x, y = elem
            img[x][y][0] = B
            img[x][y][1] = G
            img[x][y][2] = R
        minus += 1

cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img)
cv2.imwrite("./pm.bmp", img)
cv2.waitKey(0)