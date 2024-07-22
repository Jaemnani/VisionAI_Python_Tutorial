import numpy as np

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ndarr1 = np.array(['a', 'b', 'cd'])
ndarr2 = np.array(arr)
print(arr)
print(ndarr1)
print(ndarr2)

z1 = np.zeros(3)
z2 = np.zeros((3, 2))
o1 = np.ones(2)
print(z1)
print(z2)
print(o1)

x = np.arange(10)
x1 = np.arange(-1, 1.1, 0.1)
x2 = np.linspace(-1, 1, 21)

print(x)
print(x1)
print(x2)

arr = np.arange(10) # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#index 선택
print(arr[0]) # 0
print(arr[:5]) # 0, 1, 2, 3, 4
print(arr[4:]) # 4, 5, 6, 7, 8, 9
print(arr[1:4]) # 1, 2, 3
print(arr[::3]) # 0, 3, 6, 9
print(arr[1::2]) # 1, 3, 5, 7, 9
print(arr[-1]) # 9
print(arr[::-1]) # 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
print(arr[-2::-2]) # 8, 6, 4, 2, 0

a = np.array([1, 2 ,3])
b = np.array([[1, 2, 3],
						  [4, 5, 6]])
c = np.array([2, 1])
print(a + a) # 1, 4, 9
print(a - a) # 2, 4, 6
print(a * a) # 0, 0, 0
print(a / a) # 1, 1, 1
print(b + a) 
''' 2, 4, 6
	5, 7, 9 '''
print(b - a) 
''' 0, 0, 0
	3, 3, 3 '''
print(b * a)
''' 1, 4, 9
	4, 7, 9 '''
print(b / a) 
''' 1, 1, 1
	4, 2.5, 2 '''
print(a + b) 
''' 2, 4, 6
	5, 7, 9 '''
print(a - b) 
''' 0, 0, 0
	-3, -3, -3 '''
print(a * b) 
''' 1, 4, 9
	4, 7, 9 '''
print(a / b) 
''' 1, 1, 1
	0.25, 0.4, 0.5'''
print(a + 3) # 4, 5, 6

#print(a / c) # ERROR

x = np.arange(0, 3, 0.1)
y = np.sin(x)
# y = np.cos(x)
# y = np.exp(x)
# y = np.log(x)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()

print(np.round(x))
print(np.floor(x))
print(np.ceil(x))

x = np.arange(9)
print(x.shape)
print(x.reshape(3,3))
x1 = x.reshape(3,3)
print(x1.T)


print('done')