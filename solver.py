import functions
import numpy as np

def main():
    try:
        N = int(input("How many nodes?\nEnter Nodes:  "))
        n = int(input("How many time steps?\nEnter Timesteps: "))
        xi = np.linspace(0, 1, N)
        h = xi[1] - xi[0]
        dt = 1 / n
        ts = np.linspace(0, 1, n + 1)

        functions.run_finite_element_solver(N, n, xi, ts, h, dt)
    except ValueError as ve:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
