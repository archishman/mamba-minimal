import mlx.core as mx

def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
    return alpha * x + beta * y

x = mx.random.uniform(shape=(3,))
y = mx.random.uniform(shape=(3,))
a = mx.random.uniform()
b = mx.random.uniform()
print(simple_axpby(x,y,a,b))