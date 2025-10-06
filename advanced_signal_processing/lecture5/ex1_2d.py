import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 500)
plt.plot(x, 0.5*np.exp(-x/2), label='H0')
plt.plot(x, 0.25*np.exp(-x/4), label='H1')
plt.plot(x, 0.5*np.exp(-x/2)+0.25*np.exp(-x/4), label='H2')
plt.plot(x, (1/8)*np.exp(-x/8), label='H3')

plt.legend()

plt.xlabel('x')
plt.ylabel('f(x|H)')
plt.show()
