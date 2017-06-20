import cv2
import numpy
import queue

global F, img
global tmp, lost

numpy.random.seed(179)

def BFS(x, y):
    global lost
    ans = []
    F[x][y] = 1
    Q = queue.Queue()
    Q.put((x, y))
    while not Q.empty():
        x, y = Q.get()
        lost -= 1
        print(lost / (len(F) * len(F[0])))
        ans += [(x, y)]
        for dx in range(-4, 4 + 1):
            for dy in range(-4, 4 + 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[nx][ny][0] == img[x][y][0] and img[nx][ny][1] == img[x][y][1] and img[nx][ny][2] == img[x][y][2] and not F[nx][ny]:
                    Q.put((nx, ny))
                    F[nx][ny] = 1
    return ans

def nonlin(x, deriv=False):
    if deriv:
        return nonlin(x) * (1 - nonlin(x))
    return 1 / (1 + numpy.exp(-x))

def incircle(x, y, Per):
    return Per[0][0] * x ** 2 + Per[1][0] * x + Per[0][0] * y ** 2 + Per[2][0] * y + Per[3][0]

def oval(canv, X2, Y2, R, N, color = "black"):
    canv.create_oval(int((-X2 / 2 - R) * N), int((-Y2 / 2 - R) * N), int((-X2 / 2 + R) * N), int((- Y2 / 2 + R) * N), width = 2, outline = color)

def mr():
    return (-1 + numpy.random.random() * 2) / 20

Per1 = 2 * numpy.random.random((4, 1)) - 1
Per2 = 2 * numpy.random.random((4, 1)) - 1
Per3 = 2 * numpy.random.random((3, 1)) - 1
"""
Per1[0][0] = 1 + mr()
Per1[1][0] = -3/4 + mr()
Per1[2][0] = -1 + mr()
Per1[3][0] = (1/2)**2 + (3/8)**2 - (1/4)**2 + mr()

Per2[0][0] = 1 + mr()
Per2[1][0] = -5/4 + mr()
Per2[2][0] = -1 + mr()
Per2[3][0] = (1/2)**2 + (5/8)**2 - (1/4)**2 + mr()

Per3[0][0] = 1 + mr()
Per3[1][0] = 1 + mr()
Per3[2][0] = -3/2 + mr()
"""
Coeff = 100

img = cv2.imread("./pm3.jpg", cv2.IMREAD_COLOR)
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
        if S < MNW / 3.6:
            for _ in range(3):
                img[i][j][_] = 255
        else:
            for _ in range(3):
                img[i][j][_] = 0
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
            print(part)
            dy = max([elem[0] for elem in part]) - min([elem[0] for elem in part])
            dx = max([elem[1] for elem in part]) - min([elem[1] for elem in part])
            y = max([elem[0] for elem in part]) + min([elem[0] for elem in part])
            x = max([elem[1] for elem in part]) + min([elem[1] for elem in part])
            if 2 * dx >= dy and dx < 2 * dy and 8 * (dx + dy) > len(part) and len(part) > 20:
                Plus += [(y // 2, x // 2)]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        img[y // 2 + dx][x // 2 + dy][0] = 0
                        img[y // 2 + dx][x // 2 + dy][1] = 0
                        img[y // 2 + dx][x // 2 + dy][2] = 255
                """
                for elem in part:
                    img[elem[0]][elem[1]][0] = 0
                    img[elem[0]][elem[1]][1] = 0
                    img[elem[0]][elem[1]][2] = 255
                """
            elif dx >= 2 * dy and 4 * (dx + dy) > len(part) and len(part) > 10:
                Minus += [(y // 2, x // 2)]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        img[y // 2 + dx][x // 2 + dy][0] = 255
                        img[y // 2 + dx][x // 2 + dy][1] = 0
                        img[y // 2 + dx][x // 2 + dy][2] = 0
                """
                for elem in part:
                    img[elem[0]][elem[1]][0] = 255
                    img[elem[0]][elem[1]][1] = 0
                    img[elem[0]][elem[1]][2] = 0
                """
            elif len(part) < 500:
                for elem in part:
                    img[elem[0]][elem[1]][0] = 0
                    img[elem[0]][elem[1]][1] = 255
                    img[elem[0]][elem[1]][2] = 0
print("+:", len(Plus))
print("-:", len(Minus))
"""
cv2.imwrite("./newpm3.jpg", img)
exit(0)
"""

fout1 = open("plus.txt", "w")
fout2 = open("minus.txt", "w")
for elem in Plus:
    fout1.write(str(elem[0]) + " " + str(elem[1]) + "\n")
for elem in Minus:
    fout2.write(str(elem[0]) + " " + str(elem[1]) + "\n")
fout1.close()
fout2.close()
exit(0)


D = max(len(F), len(F[0]))
for i in range(len(Plus)):
    Plus[i] = (Plus[i][0] / D, Plus[i][1] / D)
for i in range(len(Minus)):
    Minus[i] = (Minus[i][0] / D, Minus[i][1] / D)
"""
Per1[0][0] = 1.05227822
Per1[1][0] = -0.74308041
Per1[2][0] = -0.91087197
Per1[3][0] = 0.44707931

Per2[0][0] = 0.84489113
Per2[1][0] = -1.437019
Per2[2][0] = -0.82743842
Per2[3][0] = 0.70742171

Per3[0][0] = 0.75024567
Per3[1][0] = 0.85316528
Per3[2][0] = -1.57330985
"""
tmp = 0
while True:
    t = numpy.random.randint(2)
    if t:
        p = numpy.random.randint(len(Plus))
        x, y = Plus[p]
    else:
        p = numpy.random.randint(len(Minus))
        x, y = Minus[p]
        t = -1
    in1 = nonlin(Coeff * incircle(x, y, Per1))
    in2 = nonlin(Coeff * incircle(x, y, Per2))
    ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
    if ans * t < 0:
        Per3[0][0] += in1 * t * 0.01
        Per3[1][0] += in2 * t * 0.01
        Per3[2][0] += t * 0.01
        d1 = nonlin(Coeff * incircle(x, y, Per1), True)
        d2 = nonlin(Coeff * incircle(x, y, Per2), True)
        coeff = t * d1 * Per3[0][0] * 0.01

        Per1[0][0] += x ** 2 * coeff
        Per1[1][0] += x * coeff
        Per1[0][0] += y ** 2 * coeff
        Per1[2][0] += y * coeff
        Per1[3][0] += coeff
        #Per1 /= numpy.sqrt(Per1[0][0] ** 2 + Per1[1][0] ** 2 + Per1[2][0] ** 2 + Per1[3][0] ** 2)
        coeff = t * d2 * Per3[1][0] * 0.01
        Per2[0][0] += x ** 2 * coeff
        Per2[1][0] += x * coeff
        Per2[0][0] += y ** 2 * coeff
        Per2[2][0] += y * coeff
        Per2[3][0] += coeff
        #Per2 /= numpy.sqrt(Per2[0][0] ** 2 + Per2[1][0] ** 2 + Per2[2][0] ** 2 + Per2[3][0] ** 2)

    tmp += 1
    print(tmp / (10000 * (len(Plus) + len(Minus))))
    if tmp > 10000 * (len(Plus) + len(Minus)):
        break

print(Per1)
print(Per2)
print(Per3)

print(len(img), len(img[0]))

ErRr = 0

tmp = 0
while True:
    t = 1
    if t:
        p = tmp
        x, y = Plus[p]
    else:
        p = numpy.random.randint(len(Minus))
        x, y = Minus[p]
        t = -1
    in1 = nonlin(Coeff * incircle(x, y, Per1))
    in2 = nonlin(Coeff * incircle(x, y, Per2))
    ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
    if ans * t < 0:
        ErRr += 1

    tmp += 1
    if tmp >= len(Plus):
        break

if len(Plus) > 0:
    print("+: ", ErRr / (len(Plus)))

ErRr = 0

tmp = 0
while True:
    t = 0
    if t:
        p = tmp
        x, y = Plus[p]
    else:
        p = tmp
        x, y = Minus[p]
        t = -1
    in1 = nonlin(Coeff * incircle(x, y, Per1))
    in2 = nonlin(Coeff * incircle(x, y, Per2))
    ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
    if ans * t < 0:
        ErRr += 1

    tmp += 1
    if tmp >= len(Minus):
        break

if len(Minus) > 0:
    print("-: ", ErRr / (len(Minus)))
"""
C, X2, Y2, R1 = Per1[0][0], Per1[1][0], Per1[2][0], Per1[3][0]

if (D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C > 0:
    X = -1 * D * X2 / 2 / C
    Y = -1 * D * Y2 / 2 / C
    R = numpy.sqrt((D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C)

    for i in range(len(F)):
        for j in range(len(F[i])):
            if abs(numpy.sqrt((i - X) ** 2 + (j - Y) ** 2) - R) < 5:
                #print("find", i, j)
                img[i][j][0] = 240
                img[i][j][1] = 240
                img[i][j][2] = 60
else:
    print("Error1")

C, X2, Y2, R1 = Per2[0][0], Per2[1][0], Per2[2][0], Per2[3][0]

if (D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C > 0:
    X = -1 * D * X2 / 2 / C
    Y = -1 * D * Y2 / 2 / C
    R = numpy.sqrt((D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C)

    for i in range(len(F)):
        for j in range(len(F[i])):
            if abs(numpy.sqrt((i - X) ** 2 + (j - Y) ** 2) - R) < 5:
                #print("find", i, j)
                img[i][j][0] = 240
                img[i][j][1] = 240
                img[i][j][2] = 60
else:
    print("Error2")
"""

img = cv2.imread("./pm3.jpg", cv2.IMREAD_COLOR)

for i in range(len(F)):
    for j in range(len(F[0])):
        x, y = i / D, j / D
        in1 = nonlin(Coeff * incircle(x, y, Per1))
        in2 = nonlin(Coeff * incircle(x, y, Per2))
        ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
        if ans > 0:
            img[i][j][0] = (240 + img[i][j][0]) // 2
            img[i][j][1] = (240 + img[i][j][1]) // 2
            img[i][j][2] = (60 + img[i][j][2]) // 2


cv2.imwrite("./newpm3.jpg", img)
