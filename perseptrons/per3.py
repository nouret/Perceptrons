import numpy
import tkinter
import time

numpy.random.seed(179)
#numpy.random.seed(57)

def nonlin(x, deriv=False):
    if deriv:
        return nonlin(x) * (1 - nonlin(x))
    return 1 / (1 + numpy.exp(-x))

def plus(x, y, t, O1, O2):
    if ((x - O1[0][0]) ** 2 + (y - O1[1][0]) ** 2 - O1[2][0] ** 2) > 0 and ((x - O2[0][0]) ** 2 + (y - O2[1][0]) ** 2 - O2[2][0] ** 2) > 0:
        return t
    return -t

def incircle(x, y, Per):
    return Per[0][0] * x ** 2 + Per[1][0] * x + Per[0][0] * y ** 2 + Per[2][0] * y + Per[3][0]

def oval(canv, X2, Y2, R, N, color = "black"):
    canv.create_oval(int((-X2 / 2 - R) * N), int((-Y2 / 2 - R) * N), int((-X2 / 2 + R) * N), int((- Y2 / 2 + R) * N), width = 2, outline = color)

def mr():
    return (-1 + numpy.random.random() * 2) / 5

O1 = numpy.array(   [[3/8],
                    [1/2],
                    [1/4]])

O2 = numpy.array(   [[5/8],
                    [1/2],
                    [1/4]])

P = []
while len(P) < 500:
    x, y, t = numpy.random.random(), numpy.random.random(), 1
    if plus(x, y, t, O1, O2) > -0.00:
        P += [(x, y, t)]

while len(P) < 1000:
    x, y, t = numpy.random.random(), numpy.random.random(), -1
    if plus(x, y, t, O1, O2) > -0.00:
        P += [(x, y, t)]

Per1 = 2 * numpy.random.random((4, 1)) - 1
Per2 = 2 * numpy.random.random((4, 1)) - 1
Per3 = 2 * numpy.random.random((3, 1)) - 1

Per1[3][0] = 1
Per2[3][0] = 1

print(Per3)

win = tkinter.Tk()
N = 500
r = 3
canv = tkinter.Canvas(win, width = N, height = N)


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


if (Per1[1][0] / 2) ** 2 + (Per1[2][0] / 2) ** 2 - Per1[3][0] > 0:
    R1 = numpy.sqrt((Per1[1][0] / 2) ** 2 + (Per1[2][0] / 2) ** 2 - Per1[3][0])
    oval(canv, Per1[1][0], Per1[2][0], R1, N, color = "green")
else:
    print("ERROR1")
if (Per2[1][0] / 2) ** 2 + (Per2[2][0] / 2) ** 2 - Per2[3][0] > 0:
    R2 = numpy.sqrt((Per2[1][0] / 2) ** 2 + (Per2[2][0] / 2) ** 2 - Per2[3][0])
    oval(canv, Per2[1][0], Per2[2][0], R2, N, color = "green")
else:
    print("ERROR2")


print(incircle(1/2, 1/2, Per1))
#exit(0)

tmp = 0
while True:
    p = numpy.random.randint(len(P))
    x, y, t = P[p]
    in1 = nonlin(100 * incircle(x, y, Per1))
    in2 = nonlin(100 * incircle(x, y, Per2))
    ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
    if ans * t < 0:
        Per3[0][0] += in1 * t * 0.01
        Per3[1][0] += in2 * t * 0.01
        Per3[2][0] += t * 0.01
        d1 = nonlin(100 * incircle(x, y, Per1), True)
        d2 = nonlin(100 * incircle(x, y, Per2), True)
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
    print(tmp)
    if tmp > 2000 * len(P):
        break


Per1 /= Per1[0][0]
Per2 /= Per2[0][0]

#print(Per1)
#print((Per1[1][0] / 2) ** 2 + (Per1[2][0] / 2) ** 2 - Per1[3][0])
#print(Per2)

if (Per1[1][0] / 2) ** 2 + (Per1[2][0] / 2) ** 2 - Per1[3][0] > 0:
    R1 = numpy.sqrt((Per1[1][0] / 2) ** 2 + (Per1[2][0] / 2) ** 2 - Per1[3][0])
    oval(canv, Per1[1][0], Per1[2][0], R1, N)
else:
    print("ERROR1")
if (Per2[1][0] / 2) ** 2 + (Per2[2][0] / 2) ** 2 - Per2[3][0] > 0:
    R2 = numpy.sqrt((Per2[1][0] / 2) ** 2 + (Per2[2][0] / 2) ** 2 - Per2[3][0])
    oval(canv, Per2[1][0], Per2[2][0], R2, N)
else:
    print("ERROR2")

#oval(canv, Per2[1][0], Per2[2][0], R2, N)

print(Per3)

for p in P:
    x, y, t = p
    in1 = nonlin(100 * incircle(x, y, Per1))
    in2 = nonlin(100 * incircle(x, y, Per2))
    ans = in1 * Per3[0][0] + in2 * Per3[1][0] + Per3[2][0]
    if ans > 0 and t > 0:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r,
            int(p[1] * N) + r, fill = "red")
    elif ans < 0 and t < 0:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r,
            int(p[1] * N) + r, fill = "blue")
    else:
        canv.create_oval(int(p[0] * N) - r, int(p[1] * N) - r, int(p[0] * N) + r,
            int(p[1] * N) + r, fill = "green")
canv.pack()
win.mainloop()