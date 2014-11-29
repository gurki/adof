import numpy as np

iter = 0
qt = 1
qdot = 0.5
lambada = max(0.0, -(2.0 * qt + 1.0) / 3.0) + 1.0

while (True): 
    iter += 1
    lambada2 = lambada * lambada
    nominator =  lambada2 * lambada
    nominator += lambada2 * (qt + 2.0)
    nominator += lambada * (2.0 * qt + 1.0)
    nominator += qt - qdot / 2.0

    denominator =  3.0 * lambada2
    denominator += 2.0 * lambada * (qt + 2.0)
    denominator += 2.0 * qt + 1.0

    dlambada = nominator / denominator

    print 'iter:', iter, ', l:', lambada, ', dl:', dlambada

    if (abs(dlambada) < 0.1):
        break

    lambada -= dlambada

print 'final result: ', lambada
