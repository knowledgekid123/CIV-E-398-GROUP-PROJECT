# beam_cli_analyser.py
import numpy as np
import matplotlib.pyplot as plt

class Beam:
    def __init__(self, length, E, I, num_elements=10):
        self.length = length
        self.E = E * 1e6   # MPa -> N/m² (SI)
        self.I = I * 1e-12 # mm⁴ -> m⁴ (SI)
        self.supports = [] # list of (node, dof) fixed
        self.loads = []    # list of loads
        self.num_elements = num_elements

    def add_support(self, pos, type_):
        node = int(round(pos / self.length * self.num_elements))
        if type_ == "pinned":
            self.supports.append((node, 0))  # vertical disp = 0
        elif type_ == "roller":
            self.supports.append((node, 0))  # vertical only
        elif type_ == "fixed":
            self.supports.append((node, 0))
            self.supports.append((node, 1))  # both disp and rotation

    def add_load(self, type_, pos1, val, pos2=None):
        self.loads.append((type_, pos1, val, pos2))

    def element_stiffness(self, L):
        """Local stiffness matrix for Euler-Bernoulli beam element"""
        EI = self.E * self.I
        k = EI / L**3 * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

    def assemble(self):
        n_nodes = self.num_elements + 1
        ndof = 2 * n_nodes
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        # Assemble global stiffness
        L = self.length / self.num_elements
        for e in range(self.num_elements):
            k = self.element_stiffness(L)
            dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += k[i, j]

        # Equivalent nodal loads
        for load in self.loads:
            typ, p1, val, p2 = load
            if typ == "point":
                node = int(round(p1 / self.length * self.num_elements))
                F[2*node] += val * 1e3  # kN -> N
            elif typ == "udl":
                start_node = int(round(p1 / self.length * self.num_elements))
                end_node = int(round(p2 / self.length * self.num_elements))
                w = val * 1e3  # kN/m -> N/m
                seg_len = (p2 - p1) / (end_node - start_node)
                for e in range(start_node, end_node):
                    Le = seg_len
                    f = w*Le/2 * np.array([1, Le/6, 1, -Le/6])
                    dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
                    for i in range(4):
                        F[dofs[i]] += f[i]

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = len(F)

        # Apply supports
        fixed = []
        for (node, dof) in self.supports:
            fixed.append(2*node + dof)

        free = [i for i in range(ndof) if i not in fixed]

        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        R = K @ d - F
        return d, R

    def postprocess(self, d):
        n_nodes = self.num_elements + 1
        x_vals = np.linspace(0, self.length, n_nodes)
        y_vals = d[0::2] * 1000  # m -> mm
        return x_vals, y_vals


def menu():
    print("\nMAIN MENU")
    print("0. Exit Program")
    print("1. Define Beam Properties")
    print("2. Add Supports")
    print("3. Add Loads")
    print("4. Run Analysis")
    print("5. Plot Results")
    print("6. Restart Program")
    return input("\nEnter option (0–6): ")


def plot_results(x, y):
    plt.figure()
    plt.plot(x, y, label="Deflection")
    plt.xlabel("Beam Length (m)")
    plt.ylabel("Deflection (mm)")
    plt.title("Deflected Shape")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    print("\n*** STRUCTURAL BEAM ANALYSER (Euler–Bernoulli FEM) ***")
    beam = None
    d = None
    x = None
    y = None

    while True:
        choice = menu()
        if choice == "0":
            break
        elif choice == "1":
            L = float(input("Beam length (m): "))
            E = float(input("E (MPa): "))
            I = float(input("I (mm^4): "))
            beam = Beam(L, E, I)
            print("Beam defined.")
        elif choice == "2":
            pos = float(input("Support position (m): "))
            typ = input("Type (pinned/roller/fixed): ")
            beam.add_support(pos, typ)
            print("Support added.")
        elif choice == "3":
            typ = input("Load type (point/udl): ")
            if typ == "point":
                pos = float(input("Position (m): "))
                val = float(input("Load (kN): "))
                beam.add_load("point", pos, val)
            elif typ == "udl":
                start = float(input("Start (m): "))
                end = float(input("End (m): "))
                val = float(input("Intensity (kN/m): "))
                beam.add_load("udl", start, val, end)
            print("Load added.")
        elif choice == "4":
            d, R = beam.solve()
            x, y = beam.postprocess(d)
            print("Analysis complete.")
            print("Reactions (N):", R)
        elif choice == "6":
            if x is not None:
                plot_results(x, y)
            else:
                print("Run analysis first.")
        elif choice == "5":
            beam = None; d = None; x = None; y = None
            print("Program Restarted.")
        else:
            print("Invalid option. Please choose a number between 0 and 6.")


if __name__ == "__main__":
    main()
