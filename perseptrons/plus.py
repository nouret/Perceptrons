import cv2
import numpy
import sys
import queue

global F, img
global tmp

sys.setrecursionlimit(50000)

def BFS(x, y):
    ans = []
    F[x][y] = 1
    Q = queue.Queue()
    Q.put((x, y))
    while not Q.empty():
        x, y = Q.get()
        ans += [(x, y)]
        for dx in [-1, 1]:
            for dy in [0]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[nx][ny][0] == img[x][y][0] and img[nx][ny][1] == img[x][y][1] and img[nx][ny][2] == img[x][y][2] and not F[nx][ny]:
                    Q.put((nx, ny))
                    F[nx][ny] = 1
        for dx in [0]:
            for dy in [-1, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[nx][ny][0] == img[x][y][0] and img[nx][ny][1] == img[x][y][1] and img[nx][ny][2] == img[x][y][2] and not F[nx][ny]:
                    Q.put((nx, ny))
                    F[nx][ny] = 1
    return ans

img = cv2.imread("./pm.bmp", cv2.IMREAD_COLOR)
F = [[0] * len(img[0]) for _ in range(len(img))]
Plus = []
Minus = []
for i in range(len(F)):
    for j in range(len(F[i])):
        if F[i][j] == 0:
            part = BFS(i, j)
            dy = max([elem[0] for elem in part]) - min([elem[0] for elem in part])
            dx = max([elem[1] for elem in part]) - min([elem[1] for elem in part])
            y = max([elem[0] for elem in part]) + min([elem[0] for elem in part])
            x = max([elem[1] for elem in part]) + min([elem[1] for elem in part])
            if dx >= dy and dx < 2 * dy and 2 * (dx + dy) > len(part):
                Plus += [(y // 2, x // 2)]
            if dx >= dy and dx >= 2 * dy and 2 * (dx + dy) > len(part):
                Minus += [(y // 2, x // 2)]
print("+:", len(Plus))
print("-:", len(Minus))