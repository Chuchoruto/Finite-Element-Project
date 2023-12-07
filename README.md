# Finite-Element-Project

This project considers the equation: u<sub>t</sub> - u<sub>xx</sub> = f(x,t) = (π<sup>2</sup>-1) * e<sup>-t</sup> sin(π*x)

The boundary conditions are u(x,0) = sin(π*x), u(0,t) = u(1,t) = 0

The analytic solution is u(x,t) = e<sup>-t</sup> sin(π*x)

## Code Overview

The code for the Finite-Element Project is structured to solve the given partial differential equation (PDE) using finite element methods (FEM). It employs both Forward and Backward Euler methods for time discretization and tackles the spatial discretization through the finite element approach. The project is organized into two primary Python scripts: `functions.py` and `solver.py`.

### `functions.py`

This script contains essential functions used in the finite element analysis. Key functionalities include:

- `compute_original_function`: Calculates the function `f(x, t) = (π^2 - 1) e^{-t} sin(π x)`.
- `find_analytical_solution`: Determines the analytical solution `u(x, t) = e^{-t} sin(π x)` of the PDE.
- `calculate_spatial_boundary`: Defines the spatial boundary condition at `t = 0`.
- `initialize_matrices`: Initializes the mass, stiffness, and force matrices used in the FEM.
- `map_local_to_global`: Creates a mapping from local to global nodes for element matrices.
- `generate_basis_functions`: Generates basis functions for the finite element method.
- `compute_matrices`: Calculates the element matrices (mass, stiffness) and force vector.
- `apply_boundary_conditions`: Applies the Dirichlet boundary conditions to the mass matrix.
- `prepare_matrices_for_euler`: Prepares matrices for Euler time integration methods.
- `solve_using_euler`: Solves the PDE using either Forward Euler or Backward Euler methods.
- `plot_solutions`: Plots both the numerical and analytical solutions for comparison.

### `solver.py`

This script is the main executable for the project. It orchestrates the finite element analysis by:

1. Prompting the user for the number of nodes (`N`) and time steps (`n`).
2. Initializing the spatial and temporal parameters based on user input.
3. Invoking the `run_finite_element_solver` function, which:
    - Asks the user to select between Forward Euler (`FE`) and Backward Euler (`BE`) methods.
    - Initializes and computes the matrices required for FEM using functions from `functions.py`.
    - Performs the numerical solution of the PDE.
    - Outputs the mass, stiffness, and force matrices, along with the numerical solution.
    - Plots the numerical solution alongside the analytical solution for comparison.

### Error Handling

The `solver.py` script includes robust error handling to ensure that the program does not terminate abruptly due to unhandled exceptions. It catches and reports invalid input and other errors, enhancing the user experience and aiding in debugging.

### Usage

To use this program, the user must:

1. Ensure that Python is installed along with the required libraries: NumPy and Matplotlib.
2. Run the `solver.py` script by calling `python3 solver.py`
3. Input the desired number of nodes and time steps when prompted.
4. Choose the preferred Euler method for time discretization.
5. View the output matrices and the plot comparing the numerical and analytical solutions.

## Project Analysis

1. Derivation of the weak form is included in the file `Weak Form Derivation.pdf`
2. Forward Euler Results
    - `Figure_1.png` shows the results of N = 11 and dt = 1/551
    - The instability occurs at dt = 1/277 as seen in `Figure_2.png`
    - The approximation undershoots the true function as N decreases as we can see with N = 9 in `Figure_3.png`
3. Backward Euler Results
    - When running the backward euler method at a timestep of dt = 1/277 we see that the solution is still stable in `Figure_4.png`. This is because backward euler method is unconditionally stable
    - If the time step becomes equal to or greater than the spatial step size, the solution becomes inaccurate because the approximation of the derivative is more rough which leads to a less accurate solution as seen in `Figure_5.png`