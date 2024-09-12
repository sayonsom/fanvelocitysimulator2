from dolfin import *
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

def simulate_airflow(fan_cmm, fan_rpm, fan_diameter, fan_position, ac_position, room_dimensions, time_duration):
    """
    Simulate the airflow inside a room based on fan and AC parameters.
    
    Parameters:
        fan_cmm (float): Fan air flow rate in Cubic Meters per Minute (CMM)
        fan_rpm (int): Fan speed in Rotations per Minute (RPM)
        fan_diameter (float): Fan diameter in meters
        fan_position (tuple): (x, y, z) coordinates of the fan
        ac_position (tuple): (x, y, z) coordinates of the AC
        room_dimensions (tuple): (length, width, height) of the room
        time_duration (float): Total time for simulation in seconds

    Returns:
        X, Y, Z, U, V, W: 3D grid coordinates and velocity components
    """
    L, W, H = room_dimensions
    fan_x, fan_y, fan_z = fan_position
    ac_x, ac_y, ac_z = ac_position

    # Create a 3D mesh with reduced resolution
    mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), 30, 15, 15)

    # Define function spaces for velocity and pressure
    V = VectorFunctionSpace(mesh, 'P', 2)  # Velocity space in 3D
    Q = FunctionSpace(mesh, 'P', 1)        # Pressure space

    # Define test and trial functions
    u = Function(V)    # Velocity field
    u0 = Function(V)   # Previous velocity field (initialized to zero)
    v = TestFunction(V)  # Test function for velocity

    # Define boundary conditions
    class InletAC(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and (1.0 <= x[1] <= 2.0) and (0.0 <= x[2] <= 2.0)

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], L) or near(x[1], W) or near(x[2], H) or near(x[2], 0))

    ac_bc = DirichletBC(V, Expression(("1.0", "0.1*sin(5.0*x[1])", "0.0"), degree=2), InletAC())
    noslip_bc = DirichletBC(V, Constant((0, 0, 0)), Walls())
    boundary_conditions = [ac_bc, noslip_bc]

    # Fan velocity calculation
    k = fan_cmm / fan_rpm  # Simplified constant
    fan_cmm_current = k * fan_rpm  # Linear relationship between CMM and RPM
    fan_area = np.pi * (fan_diameter / 2) ** 2
    fan_velocity = fan_cmm_current / (fan_area * 60)  # Velocity in m/s

    # Fan source expression
    class FanSource(UserExpression):
        def eval(self, values, x):
            r = np.sqrt((x[0] - fan_x)**2 + (x[1] - fan_y)**2 + (x[2] - fan_z)**2)
            if r > 0:
                values[0] = fan_velocity * (x[0] - fan_x) / r
                values[1] = fan_velocity * (x[1] - fan_y) / r
                values[2] = fan_velocity * (x[2] - fan_z) / r
            else:
                values[0] = values[1] = values[2] = 0.0

        def value_shape(self):
            return (3,)
    
    fan_force = FanSource(degree=2)

    # Define the Navier-Stokes equations
    rho = 1.0  # Density of air
    mu = 0.001  # Dynamic viscosity of air
    dt = 0.01  # Time step
    T = time_duration  # Total simulation time

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

    # Solver parameters
    solver_parameters = {
        'nonlinear_solver': 'newton',
        'newton_solver': {
            'linear_solver': 'gmres',  
            'preconditioner': 'ilu',
            'absolute_tolerance': 1E-6,
            'relative_tolerance': 1E-5,
            'maximum_iterations': 1000
        }
    }
    velocity_solver.parameters.update(solver_parameters)

    # Time-stepping
    t = 0.0
    n_steps = int(T / dt)
    with tqdm(total=n_steps, desc="Simulating airflow", unit="step") as pbar:
        while t < T:
            t += dt
            velocity_solver.solve()
            u0.assign(u)
            pbar.update(1)

    # Extract velocity values and interpolate for visualization
    coords = mesh.coordinates()
    u_values = u.compute_vertex_values(mesh)
    u_x = u_values[0::3]
    u_y = u_values[1::3]
    u_z = u_values[2::3]

    x_grid = np.linspace(0, L, 30)
    y_grid = np.linspace(0, W, 15)
    z_grid = np.linspace(0, H, 15)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    U = griddata(coords, u_x, (X, Y, Z), method='linear')
    V = griddata(coords, u_y, (X, Y, Z), method='linear')
    W = griddata(coords, u_z, (X, Y, Z), method='linear')

    return X, Y, Z, U, V, W
