from random import random, choice
from tkinter import *

A0, B0, C0 = 1, 1/3, -2/3

P = []
"""
P += [(1 - random() / 2, random(), 1) for i in range(500)]
P += [(random() / 2, random(), -1) for i in range(500)]
"""
while len(P) < 500:
    x, y, t = random(), random(), 1
    if t * (A0 * x + B0 * y + C0) > -0.1:
        P += [(x, y, t)]
while len(P) < 1000:
    x, y, t = random(), random(), -1
    if t * (A0 * x + B0 * y + C0) > -0.1:
        P += [(x, y, t)]

A, B, C = 0, 1, 0

find = True

tmp = 0

while True:
    p = choice(P)
    x, y, t = p
    if t * (A * x + B * y + C) <= 0:
        A += x * t * 0.01
        B += y * t * 0.01
        C += t * 0.01
        d = (A**2 + B**2 + C**2) ** 0.5
        A /= d
        B /= d
        C /= d
    tmp += 1
    if tmp > 1000 * len(P):
        break

print("tmp =", tmp)

print(A, B, C)

x1 = -C / A
x2 = (-B - C) / A

win = Tk()
N = 500
r = 3
canv = Canvas(win, width = N, height = N)
canv.create_line(int(x1 * N), 0, int(x2 * N), N, width = 2)

#print(x1, x2)

for p in P:
    if p[2] == 1:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r, int(p[1] * N) + r, fill = "red")
    else:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r, int(p[1] * N) + r, fill = "blue")
canv.pack()
win.mainloop()