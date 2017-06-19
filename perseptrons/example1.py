import numpy

# Сигмоида 
def nonlin(x, deriv=False):
    if deriv:
        return nonlin(x) * (1 - nonlin(x))
    return 1 / (1 + numpy.exp(-x))
    
# набор входных данных
X = numpy.array([   [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1]   ])

# выходные данные            
y = numpy.array([[0,0,1,1]]).T

# сделаем случайные числа более определёнными
#numpy.random.seed(1)

# инициализируем веса случайным образом со средним 0
syn0 = 2 * numpy.random.random((3,1)) - 1

for i in range(10000):

    # прямое распространение
    l0 = X
    l1 = nonlin(numpy.dot(l0, syn0))

    # насколько мы ошиблись?
    l1_error = y - l1

    # перемножим это с наклоном сигмоиды 
    # на основе значений в l1
    l1_delta = l1_error * nonlin(l1, True) # !!!

    # обновим веса
    syn0 += numpy.dot(l0.T, l1_delta) # !!!

print("Выходные данные после тренировки:")
print(l1)
