from random import random, choice
from tkinter import *

#(x - x0)^2 + (y - y0)^2 = r0^2
X0, Y0, R0 = 1/2, 1/2, 1/4

P = []

while len(P) < 500:
    x, y, t = random(), random(), 1
    if t * ((x - X0) ** 2 + (y - Y0) ** 2 - R0 ** 2) > -0.00:
        P += [(x, y, t)]
while len(P) < 1000:
    x, y, t = random(), random(), -1
    if t * ((x - X0) ** 2 + (y - Y0) ** 2 - R0 ** 2) > -0.00:
        P += [(x, y, t)]

#X1 * x^2 + X2 * x + Y1 * y^2 + Y2 * y + R1

C, X2, Y2, R1 = 1, 0, 0, 1

find = True

tmp = 0

while True:
    p = choice(P)
    x, y, t = p
    if t * (C * x ** 2 + X2 * x + C * y ** 2 + Y2 * y + R1) <= 0: #<(x^2, x, y^2, y, 1), (1, -2X, 1, -2Y, X^2 + Y^2 - R^2)>
        C += x ** 2 * t * 0.01
        X2 += x * t * 0.01
        C += y ** 2 * t * 0.01
        Y2 += y * t * 0.01
        R1 += t * 0.01
        d = (C ** 2 + X2 ** 2 + Y2 ** 2 + R1 ** 2) ** 0.5
        C /= d
        X2 /= d
        Y2 /= d
        R1 /= d
    tmp += 1
    if tmp > 1000 * len(P):
        break

print("tmp =", tmp)

X2 /= C
Y2 /= C
R1 /= C

win = Tk()
N = 500
r = 3
canv = Canvas(win, width = N, height = N)
R = ((X2 / 2) ** 2 + (Y2 / 2) ** 2 - R1) ** 0.5

canv.create_oval(int((-X2 / 2 - R) * N), int((-Y2 / 2 - R) * N), int((-X2 / 2 + R) * N), int((- Y2 / 2 + R) * N), width = 2)

#print(x1, x2)

for p in P:
    if p[2] == 1:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r, int(p[1] * N) + r, fill = "red")
    else:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r, int(p[1] * N) + r, fill = "blue")
canv.pack()
win.mainloop()