import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorednoise as cn

# def differential_equation(xn0,iterations):
#     r=3
#     K=100
#     list_of_results = []
#     dx=0
#
#     for i in range(iterations):
#
#         xn0 = xn0+dx
#         print(xn0)
#         list_of_results.append(xn0)
#         dx = r*xn0*(1-xn0/K)
#         print(dx)
#
#     return list_of_results
#
# list_of_values_5 = differential_equation(5,1000)
# list_of_values_35 = differential_equation(35,1000)
# list_of_values_70 = differential_equation(70,1000)
# list_of_values_100 = differential_equation(100,1000)
# list_of_values_120 = differential_equation(120,1000)
#
# plt.plot(list_of_values_5, label='list_of_values_5')
# # plt.plot(list_of_values_35, label='list_of_values_35')
# # plt.plot(list_of_values_70, label='list_of_values_70')
# plt.plot(list_of_values_100, label='list_of_values_100')
# plt.plot(list_of_values_120, label='list_of_values_120')
# plt.legend()
# plt.show()
time = np.arange(1,201,1)
print(time)
# pink_noise = cn.powerlaw_psd_gaussian(1,200)
# pink_noise = pink_noise - np.min(pink_noise)
# pink_noise = pink_noise * 20
# plt.plot(time,pink_noise,c='pink', zorder=1)
# plt.scatter(time,pink_noise, s=50,color='hotpink', zorder=2)
#
# plt.show()

# x = np.linspace(0, 2 * np.pi, 1000)


y = np.sin(0.5 * time)
y = y - np.min(y)
y = y * 20

plt.plot(time,y,c='red', alpha=0.7, zorder=1)
plt.scatter(time,y, s=50,color='red', zorder=2)
plt.show()
