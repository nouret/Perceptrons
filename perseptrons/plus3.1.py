import cv2
import numpy
import queue

global img
global tmp

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
