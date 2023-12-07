import numpy as np
import matplotlib.pyplot as plt

def compute_original_function(x, time):
    """
    Computes the value of the original function f(x, t).

    Parameters:
    x (float or ndarray): Spatial variable.
    time (float): Temporal variable.

    Returns:
    float or ndarray: The computed value of the original function at (x, t).
    """
    return (np.pi**2 - 1) * np.exp(-time) * np.sin(np.pi * x)

def find_analytical_solution(x, time):
    """
    Calculates the analytical solution of the differential equation at given x and time.

    Parameters:
    x (float or ndarray): Spatial variable.
    time (float): Temporal variable.

    Returns:
    float or ndarray: The analytical solution at (x, t).
    """
    return np.exp(-time) * np.sin(np.pi * x)

def calculate_spatial_boundary(x):
    """
    Defines the spatial boundary condition of the differential equation.

    Parameters:
    x (float or ndarray): Spatial variable.

    Returns:
    float or ndarray: Boundary condition at x.
    """
    return np.sin(np.pi * x)

def initialize_matrices(node_count, time_steps):
    """
    Initializes and returns the stiffness, mass, and force matrices.

    Parameters:
    node_count (int): Number of nodes in the spatial domain.
    time_steps (int): Number of time steps in the temporal domain.

    Returns:
    tuple: Tuple containing initialized mass, stiffness, and force matrices.
    """
    mass_matrix = np.zeros((node_count, node_count))
    stiffness_matrix = np.zeros((node_count, node_count))
    force_matrix = np.zeros((node_count, time_steps + 1))
    return mass_matrix, stiffness_matrix, force_matrix

def map_local_to_global(node_count):
    """
    Creates a mapping from local element indices to global indices.

    Parameters:
    node_count (int): Number of nodes in the mesh.

    Returns:
    ndarray: A 2D array where each row represents a local-to-global node mapping.
    """
    return np.vstack((np.arange(0, node_count - 1), np.arange(1, node_count))).T

def generate_basis_functions(element_length):
    """
    Generates the local basis functions and their derivatives for finite element analysis.

    Parameters:
    element_length (float): Length of the finite element.

    Returns:
    tuple: Quadrature points, basis function derivatives, and scaling factors for integrals.
    """
    # Basis function definitions
    phi1 = lambda zeta: (1 - zeta) / 2
    phi2 = lambda zeta: (1 + zeta) / 2
    basis_function_derivatives = np.array([-1 / 2, 1 / 2])
    derivative_scaling_factor = 2 / element_length
    integral_scaling_factor = element_length / 2

    # Quadrature points
    quadrature_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    # Basis functions at quadrature points
    basis_functions_at_quad = np.array([[phi1(zeta), phi2(zeta)] for zeta in quadrature_points])
    
    return basis_functions_at_quad, basis_function_derivatives, derivative_scaling_factor, integral_scaling_factor

def compute_matrices(node_count, time_steps, stiffness_matrix, mass_matrix, force_matrix, mapping, basis_functions, derivatives, derivative_scaling, integral_scaling, element_length):
    """
    Computes the element matrices (mass, stiffness) and force vector for each finite element.

    Parameters:
    node_count (int): Number of nodes in the mesh.
    time_steps (int): Number of time steps in the simulation.
    stiffness_matrix (ndarray): Global stiffness matrix.
    mass_matrix (ndarray): Global mass matrix.
    force_matrix (ndarray): Global force matrix.
    mapping (ndarray): Local-to-global node mapping.
    basis_functions (ndarray): Local basis functions.
    derivatives (ndarray): Derivatives of the basis functions.
    derivative_scaling (float): Scaling factor for derivative transformation.
    integral_scaling (float): Scaling factor for integral transformation.
    element_length (float): Length of each finite element.

    Returns:
    tuple: Updated global mass, stiffness, and force matrices.
    """
    quadrature_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    for element_index in range(node_count - 1):
        local_mass_matrix = np.zeros((2, 2))
        local_stiffness_matrix = np.zeros((2, 2))

        for i in range(2):
            for j in range(2):
                local_mass_matrix[i, j] = sum(basis_functions[i, k] * basis_functions[j, k] for k in range(2)) * element_length
                local_stiffness_matrix[i, j] = derivatives[i] * derivative_scaling * derivatives[j] * derivative_scaling * integral_scaling * 2

        global_nodes = mapping[element_index].astype(int)
        for i in range(2):
            for j in range(2):
                stiffness_matrix[global_nodes[i], global_nodes[j]] += local_stiffness_matrix[i, j]
                mass_matrix[global_nodes[i], global_nodes[j]] += local_mass_matrix[i, j]
        
        force_matrix[element_index, :] = -sum(compute_original_function(zeta, time_steps) * basis_functions[0, k] for k, zeta in enumerate(quadrature_points)) * (1/8)
        
    return mass_matrix, stiffness_matrix, force_matrix

def apply_boundary_conditions(mass_matrix, node_count):
    """
    Applies Dirichlet boundary conditions to the mass matrix.

    Parameters:
    mass_matrix (ndarray): The mass matrix to be modified.
    node_count (int): Number of nodes in the mesh.

    Returns:
    tuple: Modified mass matrix and Dirichlet boundary conditions matrix.
    """
    mass_matrix[0, :] = mass_matrix[-1, :] = mass_matrix[:, 0] = mass_matrix[:, -1] = 0
    mass_matrix[0, 0] = mass_matrix[-1, -1] = 1
    
    dirichlet_boundary_conditions = np.eye(node_count)
    dirichlet_boundary_conditions[0, 0] = dirichlet_boundary_conditions[-1, -1] = 0
    return mass_matrix, dirichlet_boundary_conditions

def prepare_matrices_for_euler(mass_matrix, stiffness_matrix, time_step):
    """
    Prepares matrices for performing Euler time integration method.

    Parameters:
    mass_matrix (ndarray): Global mass matrix.
    stiffness_matrix (ndarray): Global stiffness matrix.
    time_step (float): Time step for Euler integration.

    Returns:
    tuple: Inverse mass matrix, mass-stiffness matrix product, and inverse Euler matrix.
    """
    inverse_mass_matrix = np.linalg.inv(mass_matrix)
    mass_stiffness_product = np.dot(inverse_mass_matrix, stiffness_matrix)
    euler_matrix = (1 / time_step) * mass_matrix + stiffness_matrix
    inverse_euler_matrix = np.linalg.inv(euler_matrix)
    
    return inverse_mass_matrix, mass_stiffness_product, inverse_euler_matrix

def solve_using_euler(node_count, time_steps, time_step, mass_stiffness_product, inverse_mass_matrix, mass_matrix, force_matrix, boundary_conditions, euler_method, nodes, inverse_euler_matrix):
    """
    Solves the differential equation using Euler time integration (Forward or Backward).

    Parameters:
    node_count (int): Number of nodes in the spatial domain.
    time_steps (int): Number of time steps.
    time_step (float): Time step size.
    mass_stiffness_product (ndarray): Product of the inverse mass matrix and stiffness matrix.
    inverse_mass_matrix (ndarray): Inverse of the mass matrix.
    mass_matrix (ndarray): Mass matrix.
    force_matrix (ndarray): Force matrix.
    boundary_conditions (ndarray): Dirichlet boundary conditions.
    euler_method (str): Euler method to use ('FE' for Forward Euler, 'BE' for Backward Euler).
    nodes (ndarray): Array of node positions.
    inverse_euler_matrix (ndarray): Inverse of the Euler matrix for Backward Euler method.

    Returns:
    ndarray: Solution matrix, where each column is the solution at a time step.
    """
    solution = np.zeros((node_count, time_steps + 1))
    solution[:, 0] = calculate_spatial_boundary(nodes)

    if euler_method == 'FE':
        for t in range(time_steps):
            solution[:, t + 1] = solution[:, t] - time_step * mass_stiffness_product.dot(solution[:, t]) + time_step * inverse_mass_matrix.dot(force_matrix[:, t])
            solution[:, t + 1] = boundary_conditions.dot(solution[:, t + 1])
    else:
        for t in range(time_steps):
            solution[:, t + 1] = (1 / time_step) * inverse_euler_matrix.dot(mass_matrix.dot(solution[:, t])) + inverse_euler_matrix.dot(force_matrix[:, t])
            solution[:, t + 1] = boundary_conditions.dot(solution[:, t + 1])

    return solution

def plot_solutions(x_continuous, analytical_solution, x_discrete, numerical_solution, time_step_count, euler_method):
    """
    Plots the analytical and numerical solutions for comparison.

    Parameters:
    x_continuous (ndarray): Continuous range of x values for plotting analytical solution.
    analytical_solution (ndarray): Values of the analytical solution.
    x_discrete (ndarray): Discrete x values for plotting numerical solution.
    numerical_solution (ndarray): Values of the numerical solution at each time step.
    time_step_count (int): Time step count for labeling.
    euler_method (str): Euler method used ('FE' for Forward Euler, 'BE' for Backward Euler).

    Returns:
    None: This function plots the solutions but does not return anything.
    """
    plt.plot(x_continuous, analytical_solution, label='True Function', color = "green")
    method_label = f'Forward Euler Approximation' if euler_method == 'FE' else f'Backward Euler Approximation'
    plt.plot(x_discrete, numerical_solution[:, time_step_count], label=f'{method_label} with n = {time_step_count}', color = "red")
    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.title('True Function vs FEM Approximations')
    plt.legend()
    plt.show()


def run_finite_element_solver(N, n, xi, ts, h, dt):
    """
    Runs the finite element solver with user-selected Forward Euler or Backward Euler method.

    Parameters:
    N (int): Number of nodes.
    n (int): Number of time steps.
    xi (ndarray): Array of node positions.
    ts (ndarray): Array of time steps.
    h (float): Spatial step size.
    dt (float): Temporal step size.

    Returns:
    None: This function runs the solver and plots the results but does not return any value.
    """
    while True:
        method = input("Choose between forward euler or backward euler\nType FE or BE: ").upper()
        if method in ['FE', 'BE']:
            mass_matrix, stiffness_matrix, force_matrix = initialize_matrices(N, n)
            mapping = map_local_to_global(N)
            basis_functions, derivatives, derivative_scaling, integral_scaling = generate_basis_functions(h)
            mass_matrix, stiffness_matrix, force_matrix = compute_matrices(N, ts, stiffness_matrix, mass_matrix, force_matrix, mapping, basis_functions, derivatives, derivative_scaling, integral_scaling, h)
            mass_matrix, dirichlet_bc = apply_boundary_conditions(mass_matrix, N)
            inverse_mass_matrix, mass_stiffness_product, inverse_euler_matrix = prepare_matrices_for_euler(mass_matrix, stiffness_matrix, dt)
            solution = solve_using_euler(N, n, dt, mass_stiffness_product, inverse_mass_matrix, mass_matrix, force_matrix, dirichlet_bc, method, xi, inverse_euler_matrix)

            print('\nMass Matrix: ', mass_matrix.shape, '\n', mass_matrix)
            print('\nStiffness Matrix: ', stiffness_matrix.shape, '\n', stiffness_matrix)
            print('\nForce Matrix: ', force_matrix.shape, '\n', force_matrix)
            print('\nSolution: ', solution.shape, '\n', solution)

            x = np.linspace(0, 1, N)
            xn = np.linspace(0, 1, 1000)
            sol = find_analytical_solution(xn, 1)
            plot_solutions(xn, sol, x, solution, n, method)
            break
        else:
            print("Error: Did not Pick FE or BE")
