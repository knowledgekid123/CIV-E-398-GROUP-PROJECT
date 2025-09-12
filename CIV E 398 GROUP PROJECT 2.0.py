from operator import ne
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============ Beam FEM (Euler–Bernoulli) ============

class Beam:
    """
    Euler–Bernoulli 2D beam (vertical bending) with cubic Hermite elements:
    DOF per node: [w, theta]
    Loads supported:
      - point force at any x (converted to nearest node or split element if node exists)
      - point moment at any x (applied to nearest node if node exists)
      - UDL over [x1, x2] (consistent nodal loads on covered elements)
    """
    def __init__(self, length, E_MPa, I_mm4, num_elements=20):
        self.L = float(length)
        self.E = float(E_MPa) * 1e6        # MPa -> Pa
        self.I = float(I_mm4) * 1e-12      # mm^4 -> m^4
        self.ne = int(num_elements)
        if self.ne < 1:
            self.ne = 1

        self.x_nodes = np.linspace(0.0, self.L, self.ne + 1)
        self.supports = []   # list[(node, dof)] where dof: 0=w, 1=theta
        self.loads = []      # list of tuples describing loads
                             # ("point", x, P_kN)
                             # ("moment", x, M_kNm)
                             # ("udl", x1, x2, w_kNpm)

    # ---- Model building ----
    def add_support(self, pos_m, type_):
        node = self._nearest_node(pos_m)
        t = type_.strip().lower()
        if t == "pinned":
            self.supports.append((node, 0))          # w fixed
        elif t == "roller":
            self.supports.append((node, 0))          # w fixed
        elif t == "fixed":
            self.supports.append((node, 0))          # w fixed
            self.supports.append((node, 1))          # theta fixed
        else:
            raise ValueError("Unknown support type. Use pinned/roller/fixed.")

    def add_point_load(self, x, P_kN):
        self.loads.append(("point", float(x), float(P_kN)))

    def add_point_moment(self, x, M_kNm):
        self.loads.append(("moment", float(x), float(M_kNm)))

    def add_udl(self, x1, x2, w_kNpm):
        a, b = sorted([float(x1), float(x2)])
        if a < 0 or b > self.L or a >= b:
            raise ValueError("UDL range must be within [0, L] and x1 < x2.")
        self.loads.append(("udl", a, b, float(w_kNpm)))

    # ---- Core FEM routines ----
    def _ke(self, Le):
        EI = self.E * self.I
        L = Le
        k = EI / L**3 * np.array([
            [  12,   6*L,  -12,   6*L],
            [ 6*L, 4*L**2, -6*L, 2*L**2],
            [ -12,  -6*L,   12,  -6*L],
            [ 6*L, 2*L**2, -6*L, 4*L**2],
        ])
        return k

    def _nearest_node(self, x):
        # snap to closest mesh node
        idx = int(round((x / self.L) * self.ne))
        return max(0, min(self.ne, idx))

    def _element_span(self, e):
        x0 = self.x_nodes[e]
        x1 = self.x_nodes[e+1]
        return x0, x1, (x1 - x0)

    def assemble(self):
        ndof = 2 * (self.ne + 1)
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        # assemble stiffness
        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            ke = self._ke(Le)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += ke[i, j]

        # body loads (UDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_kNpm = vals
                w = w_kNpm * 1e3  # N/m
                # distribute to each element segment overlapped by [x1, x2]
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    # overlap length
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                    Lcov = b - a
                    # scale consistent nodal loads by (Lcov/Le)
                    fe_full = w * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])
                    scale = Lcov / Le
                    fe = fe_full * scale
                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    for i in range(4):
                        F[dofs[i]] += fe[i]

        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_kN = vals
                n = self._nearest_node(x)
                F[2*n] += P_kN * 1e3   # N (positive = upward)
            elif typ == "moment":
                x, M_kNm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_kNm * 1e3  # N·m (positive CCW)

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = K.shape[0]

        # apply supports
        fixed = sorted(set(2*node + dof for (node, dof) in self.supports))
        free = [i for i in range(ndof) if i not in fixed]

        # partition
        Kff = K[np.ix_(free, free)]
        Kfc = K[np.ix_(free, fixed)]
        Ff = F[free]

        # solve
        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        # reactions
        R_full = K @ d - F
        reactions = { (eq - 2*(eq//2), eq//2): R_full[eq] for eq in fixed }  # map to (dof, node): value

        return d, R_full, reactions

    # ---- Post-processing: w(x), M(x), V(x) ----
    # w(x) = N(ξ) d_e ;  M = EI * d2w/dx2 ; V = EI * d3w/dx3
    # Hermite cubic shape functions with ξ = x_local / Le
    @staticmethod
    def _N_hermite(ξ, L):
        N1 = 1 - 3*ξ**2 + 2*ξ**3
        N2 = L * (ξ - 2*ξ**2 + ξ**3)
        N3 = 3*ξ**2 - 2*ξ**3
        N4 = L * (-ξ**2 + ξ**3)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def _d2N_dx2(ξ, L):
        # second derivatives w.r.t. x
        # d2/dx2 = (1/L^2) d2/dξ2
        d2N1 = (-6 + 12*ξ) / (L**2)
        d2N2 = (-4 + 6*ξ) / (L)        # N2 has L factor -> (1/L^2)*(L*(-4+6ξ)) = (-4+6ξ)/L
        d2N3 = ( 6 - 12*ξ) / (L**2)
        d2N4 = (-2 + 6*ξ) / (L)        # N4 has L factor
        return np.array([d2N1, d2N2, d2N3, d2N4])

    @staticmethod
    def _d3N_dx3(ξ, L):
        # third derivatives w.r.t. x
        # d3/dx3 = (1/L^3) d3/dξ3
        d3N1 = 12 / (L**3)
        d3N2 = 6  / (L**2)   # N2 has L factor -> (1/L^3)*(6L) = 6/L^2
        d3N3 = -12 / (L**3)
        d3N4 = 6  / (L**2)   # N4 has L factor
        return np.array([d3N1, d3N2, d3N3, d3N4])

    def field_along_beam(self, d, npts_per_el=25):
        """
        Returns arrays for x, deflection(mm), moment(kN*m), shear(kN)
        """
        EI = self.E * self.I
        xs, wmm, MkNm, VkN = [], [], [], []

        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([self._N_hermite(ξ, Le)  for ξ in xi])           # (n,4)
            d2 = np.vstack([self._d2N_dx2(ξ, Le)    for ξ in xi])           # (n,4)
            d3 = np.vstack([self._d3N_dx3(ξ, Le)    for ξ in xi])           # (n,4)

            w  = (N @ de)                              # meters
            M  = EI * (d2 @ de)                        # N·m
            V  = EI * (d3 @ de)                        # N

            xs.extend(xloc.tolist())
            wmm.extend((w * 1000.0).tolist())          # mm
            MkNm.extend((M / 1e3).tolist())            # kN·m
            VkN.extend((V / 1e3).tolist())             # kN

        return np.array(xs), np.array(wmm), np.array(MkNm), np.array(VkN)

# ============ Multi-Span Beam Class ============

class MultiSpanBeam(Beam):
    def __init__(self, spans):
        self.spans = spans
        self.L = sum(span['length'] for span in spans)
        self.ne = sum(span['num_elements'] for span in spans)
        self.x_nodes = [0.0]
        self.EI = []
        for span in spans:
            Le = span['length'] / span['num_elements']
            for i in range(span['num_elements']):
                self.EI.append(span['E_MPa'] * 1e6 * span['I_mm4'] * 1e-12)  # Pa·m^4
                self.x_nodes.append(self.x_nodes[-1] + Le)
        self.x_nodes = np.array(self.x_nodes)
        self.supports = []
        self.loads = []

    def _ke(self, Le, EI):
        k = EI / Le**3 * np.array([
            [  12,   6*Le,  -12,   6*Le],
            [ 6*Le, 4*Le**2, -6*Le, 2*Le**2],
            [ -12,  -6*Le,   12,  -6*Le],
            [ 6*Le, 2*Le**2, -6*Le, 4*Le**2],
        ])
        return k

    def _element_span(self, e):
        x0 = self.x_nodes[e]
        x1 = self.x_nodes[e+1]
        return x0, x1, (x1 - x0)

    def assemble(self):
        ndof = 2 * (self.ne + 1)
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        # assemble stiffness
        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            EI = self.EI[e]
            ke = self._ke(Le, EI)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += ke[i, j]

        # body loads (UDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_kNpm = vals
                w = w_kNpm * 1e3  # N/m
                # distribute to each element segment overlapped by [x1, x2]
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    # overlap length
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                    Lcov = b - a
                    # scale consistent nodal loads by (Lcov/Le)
                    fe_full = w * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])
                    scale = Lcov / Le
                    fe = fe_full * scale
                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    for i in range(4):
                        F[dofs[i]] += fe[i]

        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_kN = vals
                n = self._nearest_node(x)
                F[2*n] += P_kN * 1e3   # N (positive = upward)
            elif typ == "moment":
                x, M_kNm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_kNm * 1e3  # N·m (positive CCW)

        return K, F

    def field_along_beam(self, d, npts_per_el=25):
        xs, wmm, MkNm, VkN = [], [], [], []
        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            EI = self.EI[e]
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([Beam._N_hermite(ξ, Le)  for ξ in xi])           # (n,4)
            d2 = np.vstack([Beam._d2N_dx2(ξ, Le)    for ξ in xi])           # (n,4)
            d3 = np.vstack([Beam._d3N_dx3(ξ, Le)    for ξ in xi])           # (n,4)

            w  = (N @ de)                              # meters
            M  = EI * (d2 @ de)                        # N·m
            V  = EI * (d3 @ de)                        # N

            xs.extend(xloc.tolist())
            wmm.extend((w * 1000.0).tolist())          # mm
            MkNm.extend((M / 1e3).tolist())            # kN·m
            VkN.extend((V / 1e3).tolist())             # kN

        return np.array(xs), np.array(wmm), np.array(MkNm), np.array(VkN)

# ============ CLI Wrapping / Plotting ============

def print_menu():
    print("\n***MAIN MENU***")
    print("0. Exit Program")
    print("1. Define Beam Properties (L, E, I, elements)")
    print("2. Add Supports (pinned/roller/fixed)")
    print("3. Add Loads (point / udl / moment)")
    print("4. Run Analysis (solve)")
    print("5. Show Reactions")
    print("6. Plot Results (Deflection, Moment, Shear)")
    print("7. Start New Beam")  # <-- updated menu text

def define_multi_span_beam():
    spans = []
    total_elements = 0
    MAX_ELEMENTS = 2000
    while True:
        print(f"\nDefining span #{len(spans)+1}:")
        length = float(input("  Length (m): "))
        E = float(input("  Young's modulus E (MPa): "))
        I = float(input("  Second moment I (mm^4): "))
        while True:
            ne = int(input(f"  Number of elements for this span (remaining allowed: {MAX_ELEMENTS - total_elements}): "))
            if 20 <= ne <= (MAX_ELEMENTS - total_elements):
                break
            else:
                print(f"  ✗ Please input a number between 20 and {MAX_ELEMENTS - total_elements}.")
        spans.append({'length': length, 'E_MPa': E, 'I_mm4': I, 'num_elements': ne})
        total_elements += ne
        if total_elements >= MAX_ELEMENTS:
            print(f"  ✓ Maximum total elements ({MAX_ELEMENTS}) reached.")
            break
        more = input("Add another span? (y/n): ").strip().lower()
        if more != "y":
            break
    return MultiSpanBeam(spans)

def add_support_cli(beam):
    while True:
        try:
            pos = float(input("Support position (m): "))
            if pos < 0 or pos > beam.L:
                print(f"  ✗ Please enter a number between 0 and {beam.L:.3f} (beam length).")
                continue
            break
        except ValueError:
            print("  ✗ Invalid input. Please enter a valid number for support position.")
    print("Select support type:")
    print("  1. Pinned")
    print("  2. Roller")
    print("  3. Fixed")
    type_map = {"1": "pinned", "2": "roller", "3": "fixed"}
    while True:
        sel = input("Type (1/2/3): ").strip()
        if sel in type_map:
            typ = type_map[sel]
            break
        else:
            print("  ✗ Invalid selection. Please enter 1, 2, or 3.")
    beam.add_support(pos, typ)
    print("  ✓ Support added at x = {:.3f} m ({})".format(pos, typ))

def add_load_cli(beam):
    print("Select load type:")
    print("  1. Point load")
    print("  2. Uniformly distributed load (UDL)")
    print("  3. Point moment")
    type_map = {"1": "point", "2": "udl", "3": "moment"}
    while True:
        sel = input("Type (1/2/3): ").strip()
        if sel in type_map:
            t = type_map[sel]
            break
        else:
            print("  ✗ Invalid selection. Please enter 1, 2, or 3.")
    if t == "point":
        while True:
            try:
                x = float(input("  Position x (m): "))
                if x < 0 or x > beam.L:
                    print(f"  ✗ Please enter a number between 0 and {beam.L:.3f} (beam length).")
                    continue
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for position.")
        while True:
            try:
                P = float(input("  Load P (kN) [downward negative]: "))
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for load.")
        beam.add_point_load(x, P)
        print("  ✓ Point load added at x = {:.3f} m: {:.3f} kN".format(x, P))
    elif t == "udl":
        while True:
            try:
                x1 = float(input("  Start x1 (m): "))
                x2 = float(input("  End   x2 (m): "))
                if x1 < 0 or x2 > beam.L or x1 >= x2:
                    print(f"  ✗ Please enter valid start/end positions within [0, {beam.L:.3f}] and x1 < x2.")
                    continue
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter valid numbers for start/end positions.")
        while True:
            try:
                w  = float(input("  Intensity w (kN/m) [downward negative]: "))
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for intensity.")
        beam.add_udl(x1, x2, w)
        print("  ✓ UDL added on [{:.3f}, {:.3f}] m: {:.3f} kN/m".format(x1, x2, w))
    elif t == "moment":
        while True:
            try:
                x = float(input("  Position x (m): "))
                if x < 0 or x > beam.L:
                    print(f"  ✗ Please enter a number between 0 and {beam.L:.3f} (beam length).")
                    continue
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for position.")
        while True:
            try:
                M = float(input("  Moment M (kN·m) [CCW positive]: "))
                break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for moment.")
        beam.add_point_moment(x, M)
        print("  ✓ Point moment added at x = {:.3f} m: {:.3f} kN·m".format(x, M))
    else:
        print("  ✗ Unknown load type.")

def plot_results(x, w_mm, M_kNm, V_kN, beam):
    import matplotlib.pyplot as plt

    # Detect multi-span and segment boundaries
    segment_boundaries = []
    if hasattr(beam, "spans"):
        cum_length = 0.0
        for span in beam.spans[:-1]:  # skip last boundary (end of beam)
            cum_length += span['length']
            segment_boundaries.append(cum_length)

    # Deflection
    plt.figure(1)
    plt.plot(x, w_mm, 'b', label="Deflection")
    plt.axhline(0, color='k', linewidth=3, label="Beam axis")  # thick black line
    for xb in segment_boundaries:
        plt.axvline(xb, color='k', linestyle='--', linewidth=2)  # segment boundary
    plt.xlabel("x (m)")
    plt.ylabel("w (mm)")
    plt.title("Deflected Shape")
    plt.grid(True)
    plt.legend()

    # Bending Moment
    plt.figure(2)
    plt.plot(x, M_kNm, 'r', label="Bending Moment")
    plt.axhline(0, color='k', linewidth=3, label="Beam axis")
    for xb in segment_boundaries:
        plt.axvline(xb, color='k', linestyle='--', linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("M (kN·m)")
    plt.title("Bending Moment Diagram (BMD)")
    plt.grid(True)
    plt.legend()

    # Shear Force
    plt.figure(3)
    plt.plot(x, V_kN, 'g', label="Shear Force")
    plt.axhline(0, color='k', linewidth=3, label="Beam axis")
    for xb in segment_boundaries:
        plt.axvline(xb, color='k', linestyle='--', linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("V (kN)")
    plt.title("Shear Force Diagram (SFD)")
    plt.grid(True)
    plt.legend()

    plt.show()

def main():
    print("\n*** STRUCTURAL BEAM ANALYSER (Euler–Bernoulli FEM, CLI) ***")
    print("Enter beam length, E (MPa), I (mm^4), supports, and loads; then solve & plot.\n")

    beam = None
    solved = False
    d = R_full = reactions = None
    x = w = M = V = None

    while True:
        print_menu()
        choice = input("\nSelect option (0–7): ").strip()

        if choice == "0":
            print("Exiting... thanks for using the Structural Beam Analyser!")
            return

        elif choice == "1":
            try:
                while True:
                    multi = input("Multi-span beam? (y/n): ").strip().lower()
                    if multi in ("y", "n"):
                        break
                    else:
                        print("  ✗ Please type 'y' or 'n'.")
                if multi == "y":
                    beam = define_multi_span_beam()
                    print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                else:
                    L = float(input("Beam length L (m): "))
                    E = float(input("Young's modulus E (MPa): "))
                    I = float(input("Second moment I (mm^4): "))
                    while True:
                        ne = int(input("Number of elements (recommend 20 — 2,000): "))
                        if 20 <= ne <= 2000:
                            break
                        else:
                            print("  ✗ Please input a number between 20 to 2000.")
                    beam = Beam(L, E, I, num_elements=ne)
                    print("  ✓ Beam created with {} elements.".format(ne))
                solved = False
            except Exception as e:
                print("  ✗ Error defining beam:", e)

        elif choice == "2":
            if beam is None:
                print("  ✗ Define the beam first (option 1).")
                continue
            try:
                while True:
                    add_support_cli(beam)
                    while True:
                        again = input("Add another support? (y/n): ").strip().lower()
                        if again == "y":
                            break  # continue outer loop to add another support
                        elif again == "n":
                            break  # or break outer loop if you want to go back to menu
                        else:
                            print("  ✗ Please type 'y' or 'n'.")
                    if again == "n":
                        break
            except Exception as e:
                print("  ✗", e)

        elif choice == "3":
            if beam is None:
                print("  ✗ Define the beam first (option 1).")
                continue
            try:
                while True:
                    add_load_cli(beam)
                    while True:
                        again = input("Add another load? (y/n): ").strip().lower()
                        if again == "y":
                            break  # continue outer loop to add another load
                        elif again == "n":
                            break  # or break outer loop if you want to go back to menu
                        else:
                            print("  ✗ Please type 'y' or 'n'.")
                    if again == "n":
                        break
            except Exception as e:
                print("  ✗", e)

        elif choice == "4":
            if beam is None:
                print("  ✗ Define the beam first (option 1).")
                continue
            try:
                start = time.time()
                d, R_full, reactions = beam.solve()
                x, w, M, V = beam.field_along_beam(d, npts_per_el=40)
                elapsed_sec = time.time() - start
                solved = True
                print(f"  ✓ Analysis complete. (Solved in {elapsed_sec:.1f} seconds)")
            except Exception as e:
                print("  ✗ Solve failed:", e)

        elif choice == "6":
            if not solved:
                print("  ✗ Run analysis first (option 4).")
                continue
            plot_results(x, w, M, V, beam)
            print("  ✓ Results plotted.")

        elif choice == "5":
            if not solved:
                print("  ✗ Run analysis first (option 4).")
                continue
            # Show only supported DOFs as reactions
            print("\n=== Reaction Forces/Moments at Supports ===")
            for (node, dof) in sorted(set(beam.supports)):
                rxn = reactions.get((dof, node), 0.0)
                x_pos = beam.x_nodes[node]
                if dof == 0:
                    print(f" x = {x_pos:.3f} m : V = {rxn/1e3:.3f} kN")
                else:
                    print(f" x = {x_pos:.3f} m : M = {rxn/1e3:.3f} kN·m")
            print("==========================================\n")

        elif choice == "7":
            if beam is not None:
                while True:
                    confirm = input("  ⚠ Are you sure you want to start a new beam? (y/n): ").strip().lower()
                    if confirm == "y":
                        beam = None
                        solved = False
                        d = R_full = reactions = None
                        x = w = M = V = None
                        print("  ✓ Model cleared. Starting new beam definition...")
                        # Jump to option 1: Define Beam Properties
                        try:
                            while True:
                                multi = input("Multi-span beam? (y/n): ").strip().lower()
                                if multi in ("y", "n"):
                                    break
                                else:
                                    print("  ✗ Please type 'y' or 'n'.")
                            if multi == "y":
                                beam = define_multi_span_beam()
                                print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                            else:
                                L = float(input("Beam length L (m): "))
                                E = float(input("Young's modulus E (MPa): "))
                                I = float(input("Second moment I (mm^4): "))
                                while True:
                                    ne = int(input("Number of elements (recommend 20 — 2,000): "))
                                    if 20 <= ne <= 2000:
                                        break
                                    else:
                                        print("  ✗ Please input a number between 20 to 2000.")
                                beam = Beam(L, E, I, num_elements=ne)
                                print("  ✓ Beam created with {} elements.".format(ne))
                            solved = False
                        except Exception as e:
                            print("  ✗ Error defining beam:", e)
                        break
                    elif confirm == "n":
                        print("  ✗ Model not cleared.")
                        break
                    else:
                        print("  ✗ Please type 'y' or 'n'.")
            else:
                print("  ⚠ No model to clear. Starting new beam definition...")
                # Jump to option 1: Define Beam Properties
                try:
                    while True:
                        multi = input("Multi-span beam? (y/n): ").strip().lower()
                        if multi in ("y", "n"):
                            break
                        else:
                            print("  ✗ Please type 'y' or 'n'.")
                    if multi == "y":
                        beam = define_multi_span_beam()
                        print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                    else:
                        L = float(input("Beam length L (m): "))
                        E = float(input("Young's modulus E (MPa): "))
                        I = float(input("Second moment I (mm^4): "))
                        while True:
                            ne = int(input("Number of elements (recommend 20 — 2,000): "))
                            if 20 <= ne <= 2000:
                                break
                            else:
                                print("  ✗ Please input a number between 20 to 2000.")
                        beam = Beam(L, E, I, num_elements=ne)
                        print("  ✓ Beam created with {} elements.".format(ne))
                    solved = False
                except Exception as e:
                    print("  ✗ Error defining beam:", e)

if __name__ == "__main__":
    main()