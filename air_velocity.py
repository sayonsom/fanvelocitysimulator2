from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# Define the 3D room geometry
L = 10.0  # Length of the room (meters)
W = 5.0   # Width of the room (meters)
H = 3.0   # Height of the room (meters)

# Create a 3D mesh with reduced resolution to avoid memory issues
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), 30, 15, 15)

# Define function spaces for velocity and pressure
V = VectorFunctionSpace(mesh, 'P', 2)  # Velocity space in 3D
Q = FunctionSpace(mesh, 'P', 1)        # Pressure space

# Define test and trial functions
u = Function(V)    # Velocity field
u0 = Function(V)   # Previous velocity field (initialized to zero)
v = TestFunction(V)  # Test function for velocity

# Define boundary conditions in 3D

# AC at the left wall (x = 0), between y = 1 and y = 2, z = 0 to z = 2
class InletAC(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and (1.0 <= x[1] <= 2.0) and (0.0 <= x[2] <= 2.0)

# No-slip boundary condition for the walls
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], L) or near(x[1], W) or near(x[2], H) or near(x[2], 0))

# Apply boundary conditions
ac_bc = DirichletBC(V, Expression(("1.0", "0.1*sin(5.0*x[1])", "0.0"), degree=2), InletAC())
noslip_bc = DirichletBC(V, Constant((0, 0, 0)), Walls())
boundary_conditions = [ac_bc, noslip_bc]

# Fan parameters
fan_rpm = 2000  # Example RPM value
fan_cmm_at_1000_rpm = 100.0  # CMM at 1000 RPM from fan specifications
fan_diameter = 0.5  # Diameter of the fan in meters

# Calculate the constant k (CMM per RPM)
k = fan_cmm_at_1000_rpm / 1000.0  # Linear approximation constant

# Calculate CMM based on the current RPM
fan_cmm = k * fan_rpm

# Calculate the cross-sectional area of the fan
fan_area = np.pi * (fan_diameter / 2) ** 2

# Calculate the velocity from the CMM value (convert CMM to m/s)
fan_velocity = fan_cmm / (fan_area * 60)

# Updated fan source expression based on RPM and calculated velocity
class FanSource(UserExpression):
    def eval(self, values, x):
        # Calculate radial distance from the fan's center
        r = np.sqrt((x[0] - 5.0)**2 + (x[1] - 2.5)**2 + (x[2] - 1.5)**2)
        if r > 0:
            values[0] = fan_velocity * (x[0] - 5.0) / r  # Radial component in x-direction
            values[1] = fan_velocity * (x[1] - 2.5) / r  # Radial component in y-direction
            values[2] = fan_velocity * (x[2] - 1.5) / r  # Radial component in z-direction
        else:
            values[0] = values[1] = values[2] = 0.0  # No airflow at the fan's center

    def value_shape(self):
        return (3,)
fan_force = FanSource(degree=2)

# Define the Navier-Stokes equations
rho = 1.0  # Density of air
mu = 0.001  # Dynamic viscosity of air

# Time-stepping parameters
dt = 0.01
T = 0.03
t = 0.0

# Define velocity residual form
F_velocity = (
    rho * dot((u - u0) / dt, v) * dx +  # Time derivative
    rho * dot(dot(u0, nabla_grad(u0)), v) * dx +  # Convection term
    mu * inner(grad(u), grad(v)) * dx -  # Diffusion term
    dot(fan_force, v) * dx  # Fan force
)

# Compute Jacobian for velocity
J_velocity = derivative(F_velocity, u)

# Create a nonlinear variational problem and solver for velocity
velocity_problem = NonlinearVariationalProblem(F_velocity, u, boundary_conditions, J_velocity)
velocity_solver = NonlinearVariationalSolver(velocity_problem)

# Switch to PETSc iterative solver
solver_parameters = {
    'nonlinear_solver': 'newton',
    'newton_solver': {
        'linear_solver': 'gmres',  # Use GMRES iterative solver
        'preconditioner': 'ilu',  # Use ILU preconditioner
        'absolute_tolerance': 1E-6,
        'relative_tolerance': 1E-5,
        'maximum_iterations': 1000
    }
}
velocity_solver.parameters.update(solver_parameters)

# Time-stepping
n_steps = int(T / dt)
with tqdm(total=n_steps, desc="Simulating airflow", unit="step") as pbar:
    while t < T:
        t += dt
        velocity_solver.solve()  # Solve the nonlinear velocity system
        u0.assign(u)  # Update u0 with the current velocity field
        pbar.update(1)

# Extract mesh coordinates and velocity values
coords = mesh.coordinates()
u_values = u.compute_vertex_values(mesh)

# Extract x, y, and z velocity components
u_x = u_values[0::3]
u_y = u_values[1::3]
u_z = u_values[2::3]

# Create the 3D grid for visualization
x_grid = np.linspace(0, L, 30)
y_grid = np.linspace(0, W, 15)
z_grid = np.linspace(0, H, 15)
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

# Interpolate velocity data to the grid for visualization
U = griddata(coords, u_x, (X, Y, Z), method='linear')
V = griddata(coords, u_y, (X, Y, Z), method='linear')
W = griddata(coords, u_z, (X, Y, Z), method='linear')

# Calculate the velocity magnitude
velocity_magnitude = np.sqrt(U**2 + V**2 + W**2)

# Create the 3D figure for streamlines
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Normalize colors based on velocity magnitude
norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
colors = plt.cm.viridis(norm(velocity_magnitude.ravel()))  # Use ravel to flatten the array

# Plot the streamlines with color based on velocity magnitude
ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color=colors)

# Add markers for the AC and Fan
# AC at (x=0, y=1.5, z=1)
ax.scatter(0, 1.5, 1.0, color='red', s=100, label='AC Inlet')

# Fan at the center of the room (x=5, y=2.5, z=1.5)
ax.scatter(5.0, 2.5, 1.5, color='green', s=100, label='Fan')

# Set labels and title
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D Airflow Streamlines with AC and Fan Positions')

# Add a color bar to represent the velocity magnitude
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(velocity_magnitude)
plt.colorbar(mappable, ax=ax, label='Velocity Magnitude')

ax.legend()

# Show the plot
plt.show()
