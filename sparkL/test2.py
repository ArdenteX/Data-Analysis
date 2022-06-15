def f(x):
    return x % 2 == 1

list(map(lambda x: x ** 2,[1,2,3,4,5]))

list(filter(f, [1, 2, 4, 6]))

import sys

def fibonacci(n):  # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n):
            return
        yield a
        a, b = b, a + b
        counter += 1


f = fibonacci(10)  # f 是一个迭代器，由生成器返回生成

while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()