#En la actividad pasada realizaste una investigación sobre el modelo del perceptrón y de cómo funciona, 
#la practica consiste en implementar el modelo en el lenguaje de programación Python. Sube el código a github
# y graba un vídeo mostrando el funcionamiento del programa, adjunta el vídeo a la actividad.

from random import choice
from numpy import array, dot, random
# que nos resuel
unit_step = lambda x: 0 if x < 0 else 1
#valores de las entradas para el perceptron 
training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]
#A continuación elegiremos tres números aleatorios entre 0 y 1 como los pesos iniciales:
w = random.rand(3)
#se utiliza para almacenar los valores de error
errors = []
# la taza de aprendisaje de la inteligencia 
eta = 0.2
#numero de iteraciones que nesecitamos 
n = 100

for i in xrange(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    #Correcion de los pesos 
    w += eta * error * x

#imprimir los resultados 
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
