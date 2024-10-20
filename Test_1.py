
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define frequency range and 1/f^(1/2) function
frequencies = np.linspace(1, 100, 1000)
y_values = 1 / frequencies
y_values = np.log10(y_values)
frequencies = np.log10(frequencies)
# Plot the graph
plt.figure(figsize=(8,6))
plt.plot(frequencies, y_values, color='blue', lw=2)
plt.title(r'Graph of $1/f^{1/2}$', fontsize=14)
plt.xlabel('Frequency (f)', fontsize=12)
plt.ylabel(r'$1/f^{1/2}$', fontsize=12)
plt.grid(True)
plt.show()





# Define the system of equations for the Shilnikov attractor
def shilnikov_attractor(t, state, a, b, c):
    x, y, z = state
    dxdt = y
    dydt = z
    dzdt = -a*z - b*y + c*(x - x**3)
    return [dxdt, dydt, dzdt]

# Parameters for the attractor
a = 0.1  # Dissipation
b = 0.1  # Coupling parameter
c = 1.0  # Nonlinearity parameter

# Initial conditions
initial_state = [1.0, 0.0, 0.0]

# Time span for the simulation
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve the system of differential equations
solution = solve_ivp(shilnikov_attractor, t_span, initial_state, args=(a, b, c), t_eval=t_eval)

# Extract the solution for each variable
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# Plotting the Shilnikov attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.6, color='blue')
ax.set_title('Shilnikov Attractor', fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Explicitly display the plot
plt.show()


