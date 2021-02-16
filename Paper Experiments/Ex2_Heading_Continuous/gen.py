import numpy as np
import Env2

start_point = []
end_point = []
for i in range(10):
    start_point.append(list(Env2.random_border_point()))
    end_point.append(list(Env2.random_inner_point()))

print(start_point)
print(end_point)


