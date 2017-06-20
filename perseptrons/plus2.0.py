import cv2
import numpy
import sys
import queue

global F, img
global tmp, lost

sys.setrecursionlimit(50000)

def BFS(x, y):
    global lost
    ans = []
    F[x][y] = 1
    Q = queue.Queue()
    Q.put((x, y))
    while not Q.empty():
        x, y = Q.get()
        lost -= 1
        print(lost)
        ans += [(x, y)]
        for dx in range(-4, 4 + 1):
            for dy in range(-4, 4 + 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[nx][ny][0] == img[x][y][0] and img[nx][ny][1] == img[x][y][1] and img[nx][ny][2] == img[x][y][2] and not F[nx][ny]:
                    Q.put((nx, ny))
                    F[nx][ny] = 1
        continue
        for dx in range(-4, 4 + 1):
            for dy in  range(-6, 6 + 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[nx][ny][0] == img[x][y][0] and img[nx][ny][1] == img[x][y][1] and img[nx][ny][2] == img[x][y][2] and not F[nx][ny]:
                    Q.put((nx, ny))
                    F[nx][ny] = 1
    return ans

img = cv2.imread("./pm2.jpg", cv2.IMREAD_COLOR)
F = [[0] * len(img[0]) for _ in range(len(img))]

Plus = []
Minus = []
MNW = 0
lost = len(F) * len(F[0])
for i in range(len(F)):
    for j in range(len(F[i])):
        MNW = max(MNW, sum([(img[i][j][_] - 255)**2 for _ in range(3)]))
print(MNW)
for i in range(len(F)):
    for j in range(len(F[i])):
        S = sum([(img[i][j][_] - 255)**2 for _ in range(3)])
        if S < MNW / 3:
            for _ in range(3):
                img[i][j][_] = 255
        else:
            for _ in range(3):
                img[i][j][_] = 0
        if (min(i, len(F) - i) / len(F) + min(j, len(F[0]) - j) / len(F[0])) < 1/3:
            for _ in range(3):
                img[i][j][_] = 255
"""
cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img)
cv2.waitKey(0)
exit(0)
"""
print("end filtre")
for i in range(len(F)):
    for j in range(len(F[i])):
        if F[i][j] == 0:
            part = BFS(i, j)
            dy = max([elem[0] for elem in part]) - min([elem[0] for elem in part])
            dx = max([elem[1] for elem in part]) - min([elem[1] for elem in part])
            y = max([elem[0] for elem in part]) + min([elem[0] for elem in part])
            x = max([elem[1] for elem in part]) + min([elem[1] for elem in part])
            if 2 * dx >= dy and dx < 2 * dy and 8 * (dx + dy) > len(part) and len(part) > 20:
                Plus += [(y // 2, x // 2)]
                for elem in part:
                    img[elem[0]][elem[1]][0] = 0
                    img[elem[0]][elem[1]][1] = 0
                    img[elem[0]][elem[1]][2] = 255
            elif dx >= 2 * dy and 4 * (dx + dy) > len(part) and len(part) > 10:
                Minus += [(y // 2, x // 2)]
                for elem in part:
                    img[elem[0]][elem[1]][0] = 255
                    img[elem[0]][elem[1]][1] = 0
                    img[elem[0]][elem[1]][2] = 0
            elif len(part) < 500:
                for elem in part:
                    img[elem[0]][elem[1]][0] = 0
                    img[elem[0]][elem[1]][1] = 255
                    img[elem[0]][elem[1]][2] = 0
print("+:", len(Plus))
print("-:", len(Minus))
#cv2.imwrite("./pm2.bmp", img)
#cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
#cv2.imshow("Display window", img)
#cv2.waitKey(0)
D = max(len(F), len(F[0]))
for i in range(len(Plus)):
    Plus[i] = (Plus[i][0] / D, Plus[i][1] / D)
for i in range(len(Minus)):
    Minus[i] = (Minus[i][0] / D, Minus[i][1] / D)

C, X2, Y2, R1 = 1, 0, 0, 1

tmp = 0

while True:
    if numpy.random.randint(0, 2) == 0:
        t = 1
        x, y = Minus[numpy.random.randint(len(Minus))]
    else:
        t = -1
        x, y = Plus[numpy.random.randint(len(Plus))]
    #print(x, y, t)
    #print(C * x ** 2 + X2 * x + C * y ** 2 + Y2 * y + R1)
    if t * (C * x ** 2 + X2 * x + C * y ** 2 + Y2 * y + R1) <= 0: #<(x^2, x, y^2, y, 1), (1, -2X, 1, -2Y, X^2 + Y^2 - R^2)>
        #print(":(")
        C += x ** 2 * t * 0.1
        X2 += x * t * 0.1
        C += y ** 2 * t * 0.1
        Y2 += y * t * 0.1
        R1 += t * 0.1
        """
        d = (C ** 2 + X2 ** 2 + Y2 ** 2 + R1 ** 2) ** 0.5
        C /= d
        X2 /= d
        Y2 /= d
        R1 /= d
        """
    tmp += 1
    print(1000 * (len(Plus) + len(Minus)) - tmp)
    if tmp > 1000 * (len(Plus) + len(Minus)):
        break

X = -1 * D * X2 / 2 / C
Y = -1 * D * Y2 / 2 / C
R = numpy.sqrt((D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C)

print("X Y R", X, Y, R)

tmp = 0
while True:
    x, y = Minus[tmp]
    t = 1
    if t * (C * x ** 2 + X2 * x + C * y ** 2 + Y2 * y + R1) <= 0:
        print(":(")
    tmp += 1
    if tmp >= len(Minus):
        break

tmp = 0
while True:
    x, y = Plus[tmp]
    t = -1
    if t * (C * x ** 2 + X2 * x + C * y ** 2 + Y2 * y + R1) <= 0:
        print(":(")
    tmp += 1
    if tmp >= len(Plus):
        break

for i in range(len(F)):
    for j in range(len(F[i])):
        if abs(numpy.sqrt((i - X) ** 2 + (j - Y) ** 2) - R) < 5:
            #print("find", i, j)
            img[i][j][0] = 240
            img[i][j][1] = 240
            img[i][j][2] = 60
cv2.imwrite("./pm2.jpg", img)
