import cv2
import numpy
import queue

global img
global tmp
global Plus, Minus

numpy.random.seed(179)

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

def count(Per1, Per2, Per3, N):
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
        if tmp > N:
            break

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

    ErRrP = ErRr
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

    return ErRrP, ErRr, Per1, Per2, Per3

def expirement(P, Const = 10000): #p1x, p1y, ..., p6x, p6y
    x1, y1, x2, y2, x3, y3 = P[:6]

    locPer1 = 2 * numpy.random.random((4, 1)) - 1
    locPer2 = 2 * numpy.random.random((4, 1)) - 1
    locPer3 = 2 * numpy.random.random((3, 1)) - 1

    locPer1[0][0] = (x2 - x3)*y1 - x1*(y2 - y3) + x3*y2 - x2*y3
    locPer1[1][0] = x1**2*(y2 - y3) + y1**2*(y2 - y3) + x2**2*y3 + y2**2*y3 - (x2**2 - x3**2 + y2**2 - y3**2)*y1 - (x3**2 + y3**2)*y2
    locPer1[2][0] = -x1**2*(x2 - x3) - x2**2*x3 - (x2 - x3)*y1**2 - x3*y2**2 + (x2**2 - x3**2 + y2**2 - y3**2)*x1 + (x3**2 + y3**2)*x2
    locPer1[3][0] = -(x3*y2 - x2*y3)*x1**2 - (x3*y2 - x2*y3)*y1**2 - (x2**2*y3 + y2**2*y3 - (x3**2 + y3**2)*y2)*x1 + (x2**2*x3 + x3*y2**2 - (x3**2 + y3**2)*x2)*y1

    locPer1 /= numpy.sqrt(locPer1[0][0] ** 2 + locPer1[1][0] ** 2 + locPer1[2][0] ** 2 + locPer1[3][0] ** 2)

    x1, y1, x2, y2, x3, y3 = P[6:]

    #by sage
    locPer2[0][0] = (x2 - x3)*y1 - x1*(y2 - y3) + x3*y2 - x2*y3
    locPer2[1][0] = x1**2*(y2 - y3) + y1**2*(y2 - y3) + x2**2*y3 + y2**2*y3 - (x2**2 - x3**2 + y2**2 - y3**2)*y1 - (x3**2 + y3**2)*y2
    locPer2[2][0] = -x1**2*(x2 - x3) - x2**2*x3 - (x2 - x3)*y1**2 - x3*y2**2 + (x2**2 - x3**2 + y2**2 - y3**2)*x1 + (x3**2 + y3**2)*x2
    locPer2[3][0] = -(x3*y2 - x2*y3)*x1**2 - (x3*y2 - x2*y3)*y1**2 - (x2**2*y3 + y2**2*y3 - (x3**2 + y3**2)*y2)*x1 + (x2**2*x3 + x3*y2**2 - (x3**2 + y3**2)*x2)*y1

    locPer2 /= numpy.sqrt(locPer2[0][0] ** 2 + locPer2[1][0] ** 2 + locPer2[2][0] ** 2 + locPer2[3][0] ** 2)

    locPer3[0][0] = 1 + mr()
    locPer3[1][0] = 1 + mr()
    locPer3[2][0] = -3/2 + mr()

    erp, erm, locPer1, locPer2, locPer3 = count(locPer1, locPer2, locPer3, Const) #big constant
    #erp, erm = 0, 0
    return (erp / len(Plus))**2 + (erm / len(Minus))**2, locPer1, locPer2, locPer3


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

Plus = []
Minus = []
fin1 = open("plus.txt", "r")
fin2 = open("minus.txt", "r")
for line in fin1.readlines():
    Plus += [tuple(map(int, line.split()))]
for line in fin2.readlines():
    Minus += [tuple(map(int, line.split()))]
fin1.close()
fin2.close()
print(len(Plus))
print(len(Minus))

if len(Plus) < 6:
    exit(0)

D = max(len(img), len(img[0]))
for i in range(len(Plus)):
    Plus[i] = (Plus[i][0] / D, Plus[i][1] / D)
for i in range(len(Minus)):
    Minus[i] = (Minus[i][0] / D, Minus[i][1] / D)

Min = 2
S = set()
while len(S) < 6:
    S |= {numpy.random.randint(len(Plus))}
S = [Plus[_] for _ in S]
MinS = S
for i in range(50): #big constant
    S = set()
    while len(S) < 6:
        S |= {numpy.random.randint(len(Plus))}
    S = [Plus[_] for _ in S]
    P = [S[0][0], S[0][1], S[1][0], S[1][1], S[4][0], S[4][1]]
    P += [S[3][0], S[3][1], S[2][0], S[2][1], S[5][0], S[5][1]]

    tmpMin, locPer1, locPer2, locPer3 = expirement(P)
    if tmpMin < Min:
        MinS = S
        Min = tmpMin

print("Min = ", Min)
S = MinS
P = [S[0][0], S[0][1], S[1][0], S[1][1], S[4][0], S[4][1]]
P += [S[3][0], S[3][1], S[2][0], S[2][1], S[5][0], S[5][1]]

tmpMin, Per1, Per2, Per3 = expirement(P, Const = 2000000)

print(tmpMin)

print("Start draw")

for i in range(len(img)):
    for j in range(len(img[0])):
        x, y = i / D, j / D
        in1 = nonlin(Coeff * incircle(x, y, Per1))
        in2 = nonlin(Coeff * incircle(x, y, Per2))
        ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
        if ans > 0:
            img[i][j][0] = (240 + img[i][j][0]) // 2
            img[i][j][1] = (240 + img[i][j][1]) // 2
            img[i][j][2] = (60 + img[i][j][2]) // 2

cv2.imwrite("./newpm3.jpg", img)

exit(0)

S = set()
while len(S) < 6:
    S |= {numpy.random.randint(len(Plus))}

for elem in S:
    x, y = Plus[elem]
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            img[x + dx][y + dy][0] = 0
            img[x + dx][y + dy][1] = 0
            img[x + dx][y + dy][2] = 255

S = [Plus[i] for i in S]

x1, y1, x2, y2, x3, y3 = S[0][0], S[0][1], S[1][0], S[1][1], S[4][0], S[4][1]

Per1[0][0] = (x2 - x3)*y1 - x1*(y2 - y3) + x3*y2 - x2*y3
Per1[1][0] = x1**2*(y2 - y3) + y1**2*(y2 - y3) + x2**2*y3 + y2**2*y3 - (x2**2 - x3**2 + y2**2 - y3**2)*y1 - (x3**2 + y3**2)*y2
Per1[2][0] = -x1**2*(x2 - x3) - x2**2*x3 - (x2 - x3)*y1**2 - x3*y2**2 + (x2**2 - x3**2 + y2**2 - y3**2)*x1 + (x3**2 + y3**2)*x2
Per1[3][0] = -(x3*y2 - x2*y3)*x1**2 - (x3*y2 - x2*y3)*y1**2 - (x2**2*y3 + y2**2*y3 - (x3**2 + y3**2)*y2)*x1 + (x2**2*x3 + x3*y2**2 - (x3**2 + y3**2)*x2)*y1

Per1 /= numpy.sqrt(Per1[0][0] ** 2 + Per1[1][0] ** 2 + Per1[2][0] ** 2 + Per1[3][0] ** 2)

x1, y1, x2, y2, x3, y3 = S[3][0], S[3][1], S[2][0], S[2][1], S[5][0], S[5][1]

#by sage
Per2[0][0] = (x2 - x3)*y1 - x1*(y2 - y3) + x3*y2 - x2*y3
Per2[1][0] = x1**2*(y2 - y3) + y1**2*(y2 - y3) + x2**2*y3 + y2**2*y3 - (x2**2 - x3**2 + y2**2 - y3**2)*y1 - (x3**2 + y3**2)*y2
Per2[2][0] = -x1**2*(x2 - x3) - x2**2*x3 - (x2 - x3)*y1**2 - x3*y2**2 + (x2**2 - x3**2 + y2**2 - y3**2)*x1 + (x3**2 + y3**2)*x2
Per2[3][0] = -(x3*y2 - x2*y3)*x1**2 - (x3*y2 - x2*y3)*y1**2 - (x2**2*y3 + y2**2*y3 - (x3**2 + y3**2)*y2)*x1 + (x2**2*x3 + x3*y2**2 - (x3**2 + y3**2)*x2)*y1

Per2 /= numpy.sqrt(Per2[0][0] ** 2 + Per2[1][0] ** 2 + Per2[2][0] ** 2 + Per2[3][0] ** 2)

print(Per1)

print(Per2)

print(len(img), len(img[0]))

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
    print(tmp / (5000 * (len(Plus) + len(Minus))))
    if tmp > 5000 * (len(Plus) + len(Minus)):
        break

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


print(Per1)
print(Per2)
print(Per3)

"""
C, X2, Y2, R1 = Per1[0][0], Per1[1][0], Per1[2][0], Per1[3][0]

if (D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C > 0:
    X = -1 * D * X2 / 2 / C
    Y = -1 * D * Y2 / 2 / C
    R = numpy.sqrt((D * X2 / C / 2)**2 + (D * Y2 / C / 2)**2 - D * D * R1 / C)

    for i in range(len(img)):
        for j in range(len(img[i])):
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

    for i in range(len(img)):
        for j in range(len(img[i])):
            if abs(numpy.sqrt((i - X) ** 2 + (j - Y) ** 2) - R) < 5:
                #print("find", i, j)
                img[i][j][0] = 240
                img[i][j][1] = 240
                img[i][j][2] = 60
else:
    print("Error2")
"""

for i in range(len(img)):
    print(len(img) - i)
    for j in range(len(img[0])):
        x, y = i / D, j / D
        in1 = nonlin(Coeff * incircle(x, y, Per1))
        in2 = nonlin(Coeff * incircle(x, y, Per2))
        ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
        if ans > 0:
            img[i][j][0] = (240 + img[i][j][0]) // 2
            img[i][j][1] = (240 + img[i][j][1]) // 2
            img[i][j][2] = (60 + img[i][j][2]) // 2


cv2.imwrite("./newpm3.jpg", img)
