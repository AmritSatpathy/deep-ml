import numpy
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    return numpy.array(a)@numpy.array(b) if numpy.array(a).shape[-1] == numpy.array(b).shape[-1] else -1