import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============ Beam FEM (Euler–Bernoulli) ============

class Beam:
    """
    Euler–Bernoulli 2D beam (vertical bending) with cubic Hermite elements.
    Supports two unit systems: N-mm-MPa or kN-m-kPa. All internal calculations
    are converted to SI (N, m, Pa) for consistency.
    DOF per node: [w, theta]
    Loads supported: point force, point moment, and UDL.
    """
    def __init__(self, length, E_val, I_val, h_val, num_elements=20, unit_system='kN-m-kPa'):
        self.ne = int(num_elements)
        if self.ne < 1:
            self.ne = 1

        # Store the chosen unit system
        self.unit_system = unit_system
        
        # Internally, we work with SI units (m, Pa, N)
        if self.unit_system == 'N-mm-MPa':
            self.L = float(length) / 1000.0        # mm -> m
            self.E = float(E_val) * 1e6            # MPa -> Pa
            self.I = float(I_val) * 1e-12          # mm^4 -> m^4
            self.h = float(h_val) / 1000.0         # mm -> m
        elif self.unit_system == 'kN-m-kPa':
            self.L = float(length)                 # m -> m
            self.E = float(E_val) * 1e3            # kPa -> Pa
            self.I = float(I_val)                  # m^4 -> m^4
            self.h = float(h_val)                  # m -> m
        else:
            raise ValueError("Invalid unit system. Use 'N-mm-MPa' or 'kN-m-kPa'.")

        self.x_nodes = np.linspace(0.0, self.L, self.ne + 1)
        self.supports = []   # list[(node, dof)] where dof: 0=w, 1=theta
        self.loads = []      # list of tuples describing loads
                             # ("point", x_m, P_N)
                             # ("moment", x_m, M_Nm)
                             # ("udl", x1_m, x2_m, w_N_m)
                             # ("lvdl", x1_m, x2_m, w1_N_m, w2_N_m)

    # ---- Model building ----
    def add_support(self, pos_val, type_):
        # Position is converted to meters internally
        pos_m = pos_val / 1000.0 if self.unit_system == 'N-mm-MPa' else pos_val
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

    def add_point_load(self, x_val, P_val):
        # Convert position and load to SI units
        x_m = x_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x_val
        P_N = P_val if self.unit_system == 'N-mm-MPa' else P_val * 1e3
        self.loads.append(("point", float(x_m), float(P_N)))

    def add_point_moment(self, x_val, M_val):
        # Convert position and moment to SI units
        x_m = x_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x_val
        M_Nm = M_val if self.unit_system == 'N-mm-MPa' else M_val * 1e3
        self.loads.append(("moment", float(x_m), float(M_Nm)))

    def add_udl(self, x1_val, x2_val, w_val):
        # Convert positions and load to SI units
        x1_m = x1_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x1_val
        x2_m = x2_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x2_val
        w_N_m = w_val if self.unit_system == 'N-mm-MPa' else w_val * 1e3
        a, b = sorted([float(x1_m), float(x2_m)])
        if a < 0 or b > self.L or a >= b:
            raise ValueError(f"UDL range must be within [0, {self.L:.3f}] and x1 < x2.")
        self.loads.append(("udl", a, b, float(w_N_m)))
        
    def add_lvdl(self, x1_val, x2_val, w1_val, w2_val):
        # Convert positions and loads to SI units
        x1_m = x1_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x1_val
        x2_m = x2_val / 1000.0 if self.unit_system == 'N-mm-MPa' else x2_val
        w1_N_m = w1_val if self.unit_system == 'N-mm-MPa' else w1_val * 1e3
        w2_N_m = w2_val if self.unit_system == 'N-mm-MPa' else w2_val * 1e3
        a, b = sorted([float(x1_m), float(x2_m)])
        if a < 0 or b > self.L or a >= b:
            raise ValueError(f"LVDL range must be within [0, {self.L:.3f}] and x1 < x2.")
        self.loads.append(("lvdl", a, b, float(w1_N_m), float(w2_N_m)))

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

        # body loads (UDL and LVDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_N_m = vals
                w = w_N_m  # N/m
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

            elif typ == "lvdl":
                x1, x2, w1_N_m, w2_N_m = vals
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    
                    # check for element overlap with LVDL span
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                        
                    # Calculate w at element ends
                    # linear interpolation: w(x) = w1 + (w2-w1)/(x2-x1) * (x-x1)
                    w_load_span = x2 - x1
                    w_at_start = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (a - x1)
                    w_at_end = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (b - x1)
                    
                    # Consistent nodal forces for trapezoidal load
                    fe_v1 = (b-a) / 6 * (2*w_at_start + w_at_end)
                    fe_m1 = (b-a)**2 / 24 * (3*w_at_start + w_at_end)
                    fe_v2 = (b-a) / 6 * (w_at_start + 2*w_at_end)
                    fe_m2 = (b-a)**2 / 24 * (w_at_start + 3*w_at_end)

                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    F[dofs[0]] += fe_v1
                    F[dofs[1]] += fe_m1
                    F[dofs[2]] += fe_v2
                    F[dofs[3]] -= fe_m2 # Note the sign change for moment at right node
        
        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_N = vals
                n = self._nearest_node(x)
                F[2*n] += P_N    # N (positive = upward)
            elif typ == "moment":
                x, M_Nm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_Nm # N·m (positive CCW)

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = K.shape[0]

        # apply supports
        fixed = sorted(set(2*node + dof for (node, dof) in self.supports))
        free = [i for i in range(ndof) if i not in fixed]

        # partition
        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        # solve
        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        # reactions
        R_full = K @ d - F
        reactions = { (eq - 2*(eq//2), eq//2): R_full[eq] for eq in fixed }

        return d, R_full, reactions

    # ---- Post-processing: w(x), M(x), V(x) ----
    @staticmethod
    def _N_hermite(ξ, L):
        N1 = 1 - 3*ξ**2 + 2*ξ**3
        N2 = L * (ξ - 2*ξ**2 + ξ**3)
        N3 = 3*ξ**2 - 2*ξ**3
        N4 = L * (-ξ**2 + ξ**3)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def _d2N_dx2(ξ, L):
        d2N1 = (-6 + 12*ξ) / (L**2)
        d2N2 = (-4 + 6*ξ) / (L)
        d2N3 = ( 6 - 12*ξ) / (L**2)
        d2N4 = (-2 + 6*ξ) / (L)
        return np.array([d2N1, d2N2, d2N3, d2N4])

    @staticmethod
    def _d3N_dx3(ξ, L):
        d3N1 = 12 / (L**3)
        d3N2 = 6  / (L**2)
        d3N3 = -12 / (L**3)
        d3N4 = 6  / (L**2)
        return np.array([d3N1, d3N2, d3N3, d3N4])

    def field_along_beam(self, d, npts_per_el=25):
        """
        Returns arrays for x, deflection, moment, shear, and stress in the chosen units.
        """
        xs, w_si, M_si, V_si = [], [], [], []

        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([self._N_hermite(ξ, Le)  for ξ in xi])
            d2 = np.vstack([self._d2N_dx2(ξ, Le)    for ξ in xi])
            d3 = np.vstack([self._d3N_dx3(ξ, Le)    for ξ in xi])

            w  = (N @ de)                              # m
            M  = self.E * self.I * (d2 @ de)           # N·m
            V  = self.E * self.I * (d3 @ de)           # N

            xs.extend(xloc.tolist())
            w_si.extend(w.tolist())
            M_si.extend(M.tolist())
            V_si.extend(V.tolist())

        # Convert results to the specified output units
        xs_out = np.array(xs)
        w_out = np.array(w_si)
        M_out = np.array(M_si)
        V_out = np.array(V_si)
        
        # Calculate stress: sigma = M*y/I
        # y_top is h/2, y_bot is -h/2
        sigma_top_si = M_out * (self.h / 2.0) / self.I
        sigma_bot_si = M_out * (-self.h / 2.0) / self.I

        if self.unit_system == 'N-mm-MPa':
            x_out = xs_out * 1000.0          # m -> mm
            w_out = w_out * 1000.0           # m -> mm
            M_out = M_out * 1000.0           # N·m -> N·mm
            V_out = V_out                    # N -> N
            sigma_top_out = sigma_top_si / 1e6 # Pa -> MPa
            sigma_bot_out = sigma_bot_si / 1e6 # Pa -> MPa
        elif self.unit_system == 'kN-m-kPa':
            x_out = xs_out                   # m -> m
            w_out = w_out * 1000.0           # m -> mm (displaying in mm for readability)
            M_out = M_out / 1e3              # N·m -> kN·m
            V_out = V_out / 1e3              # N -> kN
            sigma_top_out = sigma_top_si / 1e3 # Pa -> kPa
            sigma_bot_out = sigma_bot_si / 1e3 # Pa -> kPa
        
        return x_out, w_out, M_out, V_out, sigma_top_out, sigma_bot_out

# ============ Multi-Span Beam Class ============

class MultiSpanBeam(Beam):
    def __init__(self, spans, unit_system='kN-m-kPa'):
        self.spans = spans
        self.unit_system = unit_system
        
        # Internally, we work in SI units (m, Pa, N)
        if self.unit_system == 'N-mm-MPa':
            self.L = sum(span['length'] for span in spans) / 1000.0
        else:
            self.L = sum(span['length'] for span in spans)
        
        self.ne = sum(span['num_elements'] for span in spans)
        
        self.x_nodes = [0.0]
        self.EI = []
        self.h = []
        
        for span in spans:
            # Convert span properties to SI
            if self.unit_system == 'N-mm-MPa':
                Le = (span['length'] / 1000.0) / span['num_elements']
                E_si = span['E_val'] * 1e6
                I_si = span['I_val'] * 1e-12
                h_si = span['h_val'] / 1000.0
            else: # kN-m-kPa
                Le = span['length'] / span['num_elements']
                E_si = span['E_val'] * 1e3
                I_si = span['I_val']
                h_si = span['h_val']
            
            for i in range(span['num_elements']):
                self.EI.append(E_si * I_si)
                self.h.append(h_si)
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

        # body loads (UDL and LVDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_N_m = vals
                w = w_N_m  # N/m
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                    Lcov = b - a
                    fe_full = w * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])
                    scale = Lcov / Le
                    fe = fe_full * scale
                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    for i in range(4):
                        F[dofs[i]] += fe[i]

            elif typ == "lvdl":
                x1, x2, w1_N_m, w2_N_m = vals
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    
                    # check for element overlap with LVDL span
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                        
                    # Calculate w at element ends
                    w_load_span = x2 - x1
                    w_at_start = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (a - x1)
                    w_at_end = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (b - x1)
                    
                    # Consistent nodal forces for trapezoidal load
                    fe_v1 = (b-a) / 6 * (2*w_at_start + w_at_end)
                    fe_m1 = (b-a)**2 / 24 * (3*w_at_start + w_at_end)
                    fe_v2 = (b-a) / 6 * (w_at_start + 2*w_at_end)
                    fe_m2 = (b-a)**2 / 24 * (w_at_start + 3*w_at_end)

                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    F[dofs[0]] += fe_v1
                    F[dofs[1]] += fe_m1
                    F[dofs[2]] += fe_v2
                    F[dofs[3]] -= fe_m2 # Note the sign change for moment at right node
        
        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_N = vals
                n = self._nearest_node(x)
                F[2*n] += P_N    # N (positive = upward)
            elif typ == "moment":
                x, M_Nm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_Nm # N·m (positive CCW)

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = K.shape[0]

        # apply supports
        fixed = sorted(set(2*node + dof for (node, dof) in self.supports))
        free = [i for i in range(ndof) if i not in fixed]

        # partition
        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        # solve
        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        # reactions
        R_full = K @ d - F
        reactions = { (eq - 2*(eq//2), eq//2): R_full[eq] for eq in fixed }

        return d, R_full, reactions

    # ---- Post-processing: w(x), M(x), V(x) ----
    @staticmethod
    def _N_hermite(ξ, L):
        N1 = 1 - 3*ξ**2 + 2*ξ**3
        N2 = L * (ξ - 2*ξ**2 + ξ**3)
        N3 = 3*ξ**2 - 2*ξ**3
        N4 = L * (-ξ**2 + ξ**3)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def _d2N_dx2(ξ, L):
        d2N1 = (-6 + 12*ξ) / (L**2)
        d2N2 = (-4 + 6*ξ) / (L)
        d2N3 = ( 6 - 12*ξ) / (L**2)
        d2N4 = (-2 + 6*ξ) / (L)
        return np.array([d2N1, d2N2, d2N3, d2N4])

    @staticmethod
    def _d3N_dx3(ξ, L):
        d3N1 = 12 / (L**3)
        d3N2 = 6  / (L**2)
        d3N3 = -12 / (L**3)
        d3N4 = 6  / (L**2)
        return np.array([d3N1, d3N2, d3N3, d3N4])

    def field_along_beam(self, d, npts_per_el=25):
        """
        Returns arrays for x, deflection, moment, shear, and stress in the chosen units.
        """
        xs, w_si, M_si, V_si = [], [], [], []

        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([self._N_hermite(ξ, Le)  for ξ in xi])
            d2 = np.vstack([self._d2N_dx2(ξ, Le)    for ξ in xi])
            d3 = np.vstack([self._d3N_dx3(ξ, Le)    for ξ in xi])

            w  = (N @ de)                              # m
            M  = self.E * self.I * (d2 @ de)           # N·m
            V  = self.E * self.I * (d3 @ de)           # N

            xs.extend(xloc.tolist())
            w_si.extend(w.tolist())
            M_si.extend(M.tolist())
            V_si.extend(V.tolist())

        # Convert results to the specified output units
        xs_out = np.array(xs)
        w_out = np.array(w_si)
        M_out = np.array(M_si)
        V_out = np.array(V_si)
        
        # Calculate stress: sigma = M*y/I
        # y_top is h/2, y_bot is -h/2
        sigma_top_si = M_out * (self.h / 2.0) / self.I
        sigma_bot_si = M_out * (-self.h / 2.0) / self.I

        if self.unit_system == 'N-mm-MPa':
            x_out = xs_out * 1000.0          # m -> mm
            w_out = w_out * 1000.0           # m -> mm
            M_out = M_out * 1000.0           # N·m -> N·mm
            V_out = V_out                    # N -> N
            sigma_top_out = sigma_top_si / 1e6 # Pa -> MPa
            sigma_bot_out = sigma_bot_si / 1e6 # Pa -> MPa
        elif self.unit_system == 'kN-m-kPa':
            x_out = xs_out                   # m -> m
            w_out = w_out * 1000.0           # m -> mm (displaying in mm for readability)
            M_out = M_out / 1e3              # N·m -> kN·m
            V_out = V_out / 1e3              # N -> kN
            sigma_top_out = sigma_top_si / 1e3 # Pa -> kPa
            sigma_bot_out = sigma_bot_si / 1e3 # Pa -> kPa
        
        return x_out, w_out, M_out, V_out, sigma_top_out, sigma_bot_out

# ============ Multi-Span Beam Class ============

class MultiSpanBeam(Beam):
    def __init__(self, spans, unit_system='kN-m-kPa'):
        self.spans = spans
        self.unit_system = unit_system
        
        # Internally, we work in SI units (m, Pa, N)
        if self.unit_system == 'N-mm-MPa':
            self.L = sum(span['length'] for span in spans) / 1000.0
        else:
            self.L = sum(span['length'] for span in spans)
        
        self.ne = sum(span['num_elements'] for span in spans)
        
        self.x_nodes = [0.0]
        self.EI = []
        self.h = []
        
        for span in spans:
            # Convert span properties to SI
            if self.unit_system == 'N-mm-MPa':
                Le = (span['length'] / 1000.0) / span['num_elements']
                E_si = span['E_val'] * 1e6
                I_si = span['I_val'] * 1e-12
                h_si = span['h_val'] / 1000.0
            else: # kN-m-kPa
                Le = span['length'] / span['num_elements']
                E_si = span['E_val'] * 1e3
                I_si = span['I_val']
                h_si = span['h_val']
            
            for i in range(span['num_elements']):
                self.EI.append(E_si * I_si)
                self.h.append(h_si)
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

        # body loads (UDL and LVDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_N_m = vals
                w = w_N_m  # N/m
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                    Lcov = b - a
                    fe_full = w * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])
                    scale = Lcov / Le
                    fe = fe_full * scale
                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    for i in range(4):
                        F[dofs[i]] += fe[i]

            elif typ == "lvdl":
                x1, x2, w1_N_m, w2_N_m = vals
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    
                    # check for element overlap with LVDL span
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                        
                    # Calculate w at element ends
                    w_load_span = x2 - x1
                    w_at_start = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (a - x1)
                    w_at_end = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (b - x1)
                    
                    # Consistent nodal forces for trapezoidal load
                    fe_v1 = (b-a) / 6 * (2*w_at_start + w_at_end)
                    fe_m1 = (b-a)**2 / 24 * (3*w_at_start + w_at_end)
                    fe_v2 = (b-a) / 6 * (w_at_start + 2*w_at_end)
                    fe_m2 = (b-a)**2 / 24 * (w_at_start + 3*w_at_end)

                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    F[dofs[0]] += fe_v1
                    F[dofs[1]] += fe_m1
                    F[dofs[2]] += fe_v2
                    F[dofs[3]] -= fe_m2 # Note the sign change for moment at right node
        
        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_N = vals
                n = self._nearest_node(x)
                F[2*n] += P_N    # N (positive = upward)
            elif typ == "moment":
                x, M_Nm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_Nm # N·m (positive CCW)

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = K.shape[0]

        # apply supports
        fixed = sorted(set(2*node + dof for (node, dof) in self.supports))
        free = [i for i in range(ndof) if i not in fixed]

        # partition
        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        # solve
        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        # reactions
        R_full = K @ d - F
        reactions = { (eq - 2*(eq//2), eq//2): R_full[eq] for eq in fixed }

        return d, R_full, reactions

    # ---- Post-processing: w(x), M(x), V(x) ----
    @staticmethod
    def _N_hermite(ξ, L):
        N1 = 1 - 3*ξ**2 + 2*ξ**3
        N2 = L * (ξ - 2*ξ**2 + ξ**3)
        N3 = 3*ξ**2 - 2*ξ**3
        N4 = L * (-ξ**2 + ξ**3)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def _d2N_dx2(ξ, L):
        d2N1 = (-6 + 12*ξ) / (L**2)
        d2N2 = (-4 + 6*ξ) / (L)
        d2N3 = ( 6 - 12*ξ) / (L**2)
        d2N4 = (-2 + 6*ξ) / (L)
        return np.array([d2N1, d2N2, d2N3, d2N4])

    @staticmethod
    def _d3N_dx3(ξ, L):
        d3N1 = 12 / (L**3)
        d3N2 = 6  / (L**2)
        d3N3 = -12 / (L**3)
        d3N4 = 6  / (L**2)
        return np.array([d3N1, d3N2, d3N3, d3N4])

    def field_along_beam(self, d, npts_per_el=25):
        """
        Returns arrays for x, deflection, moment, shear, and stress in the chosen units.
        """
        xs, w_si, M_si, V_si = [], [], [], []

        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([self._N_hermite(ξ, Le)  for ξ in xi])
            d2 = np.vstack([self._d2N_dx2(ξ, Le)    for ξ in xi])
            d3 = np.vstack([self._d3N_dx3(ξ, Le)    for ξ in xi])

            w  = (N @ de)                              # m
            M  = self.E * self.I * (d2 @ de)           # N·m
            V  = self.E * self.I * (d3 @ de)           # N

            xs.extend(xloc.tolist())
            w_si.extend(w.tolist())
            M_si.extend(M.tolist())
            V_si.extend(V.tolist())

        # Convert results to the specified output units
        xs_out = np.array(xs)
        w_out = np.array(w_si)
        M_out = np.array(M_si)
        V_out = np.array(V_si)
        
        # Calculate stress: sigma = M*y/I
        # y_top is h/2, y_bot is -h/2
        sigma_top_si = M_out * (self.h / 2.0) / self.I
        sigma_bot_si = M_out * (-self.h / 2.0) / self.I

        if self.unit_system == 'N-mm-MPa':
            x_out = xs_out * 1000.0          # m -> mm
            w_out = w_out * 1000.0           # m -> mm
            M_out = M_out * 1000.0           # N·m -> N·mm
            V_out = V_out                    # N -> N
            sigma_top_out = sigma_top_si / 1e6 # Pa -> MPa
            sigma_bot_out = sigma_bot_si / 1e6 # Pa -> MPa
        elif self.unit_system == 'kN-m-kPa':
            x_out = xs_out                   # m -> m
            w_out = w_out * 1000.0           # m -> mm (displaying in mm for readability)
            M_out = M_out / 1e3              # N·m -> kN·m
            V_out = V_out / 1e3              # N -> kN
            sigma_top_out = sigma_top_si / 1e3 # Pa -> kPa
            sigma_bot_out = sigma_bot_si / 1e3 # Pa -> kPa
        
        return x_out, w_out, M_out, V_out, sigma_top_out, sigma_bot_out

# ============ Multi-Span Beam Class ============

class MultiSpanBeam(Beam):
    def __init__(self, spans, unit_system='kN-m-kPa'):
        self.spans = spans
        self.unit_system = unit_system
        
        # Internally, we work in SI units (m, Pa, N)
        if self.unit_system == 'N-mm-MPa':
            self.L = sum(span['length'] for span in spans) / 1000.0
        else:
            self.L = sum(span['length'] for span in spans)
        
        self.ne = sum(span['num_elements'] for span in spans)
        
        self.x_nodes = [0.0]
        self.EI = []
        self.h = []
        
        for span in spans:
            # Convert span properties to SI
            if self.unit_system == 'N-mm-MPa':
                Le = (span['length'] / 1000.0) / span['num_elements']
                E_si = span['E_val'] * 1e6
                I_si = span['I_val'] * 1e-12
                h_si = span['h_val'] / 1000.0
            else: # kN-m-kPa
                Le = span['length'] / span['num_elements']
                E_si = span['E_val'] * 1e3
                I_si = span['I_val']
                h_si = span['h_val']
            
            for i in range(span['num_elements']):
                self.EI.append(E_si * I_si)
                self.h.append(h_si)
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

        # body loads (UDL and LVDL) to equivalent nodal loads
        for typ, *vals in self.loads:
            if typ == "udl":
                x1, x2, w_N_m = vals
                w = w_N_m  # N/m
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                    Lcov = b - a
                    fe_full = w * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])
                    scale = Lcov / Le
                    fe = fe_full * scale
                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    for i in range(4):
                        F[dofs[i]] += fe[i]

            elif typ == "lvdl":
                x1, x2, w1_N_m, w2_N_m = vals
                for e in range(self.ne):
                    ex0, ex1, Le = self._element_span(e)
                    
                    # check for element overlap with LVDL span
                    a = max(ex0, x1)
                    b = min(ex1, x2)
                    if b <= a:
                        continue
                        
                    # Calculate w at element ends
                    w_load_span = x2 - x1
                    w_at_start = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (a - x1)
                    w_at_end = w1_N_m + (w2_N_m - w1_N_m) / w_load_span * (b - x1)
                    
                    # Consistent nodal forces for trapezoidal load
                    fe_v1 = (b-a) / 6 * (2*w_at_start + w_at_end)
                    fe_m1 = (b-a)**2 / 24 * (3*w_at_start + w_at_end)
                    fe_v2 = (b-a) / 6 * (w_at_start + 2*w_at_end)
                    fe_m2 = (b-a)**2 / 24 * (w_at_start + 3*w_at_end)

                    dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                    F[dofs[0]] += fe_v1
                    F[dofs[1]] += fe_m1
                    F[dofs[2]] += fe_v2
                    F[dofs[3]] -= fe_m2 # Note the sign change for moment at right node
        
        # nodal point loads / moments (snap to nearest node)
        for typ, *vals in self.loads:
            if typ == "point":
                x, P_N = vals
                n = self._nearest_node(x)
                F[2*n] += P_N
            elif typ == "moment":
                x, M_Nm = vals
                n = self._nearest_node(x)
                F[2*n + 1] += M_Nm

        return K, F

    def solve(self):
        K, F = self.assemble()
        ndof = K.shape[0]

        # apply supports
        fixed = sorted(set(2*node + dof for (node, dof) in self.supports))
        free = [i for i in range(ndof) if i not in fixed]

        # partition
        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        # solve
        d = np.zeros(ndof)
        d[free] = np.linalg.solve(Kff, Ff)

        # reactions
        R_full = K @ d - F
        reactions = { (eq - 2*(eq//2), eq//2): R_full[eq] for eq in fixed }

        return d, R_full, reactions

    # ---- Post-processing: w(x), M(x), V(x) ----
    @staticmethod
    def _N_hermite(ξ, L):
        N1 = 1 - 3*ξ**2 + 2*ξ**3
        N2 = L * (ξ - 2*ξ**2 + ξ**3)
        N3 = 3*ξ**2 - 2*ξ**3
        N4 = L * (-ξ**2 + ξ**3)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def _d2N_dx2(ξ, L):
        d2N1 = (-6 + 12*ξ) / (L**2)
        d2N2 = (-4 + 6*ξ) / (L)
        d2N3 = ( 6 - 12*ξ) / (L**2)
        d2N4 = (-2 + 6*ξ) / (L)
        return np.array([d2N1, d2N2, d2N3, d2N4])

    @staticmethod
    def _d3N_dx3(ξ, L):
        d3N1 = 12 / (L**3)
        d3N2 = 6  / (L**2)
        d3N3 = -12 / (L**3)
        d3N4 = 6  / (L**2)
        return np.array([d3N1, d3N2, d3N3, d3N4])

    def field_along_beam(self, d, npts_per_el=25):
        """
        Returns arrays for x, deflection, moment, shear, and stress in the chosen units.
        """
        xs, w_si, M_si, V_si = [], [], [], []

        for e in range(self.ne):
            x0, x1, Le = self._element_span(e)
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            de = d[dofs]

            xi = np.linspace(0, 1, npts_per_el, endpoint=(e == self.ne-1))
            xloc = x0 + xi * Le

            N  = np.vstack([self._N_hermite(ξ, Le)  for ξ in xi])
            d2 = np.vstack([self._d2N_dx2(ξ, Le)    for ξ in xi])
            d3 = np.vstack([self._d3N_dx3(ξ, Le)    for ξ in xi])

            w  = (N @ de)                              # m
            M  = self.E * self.I * (d2 @ de)           # N·m
            V  = self.E * self.I * (d3 @ de)           # N

            xs.extend(xloc.tolist())
            w_si.extend(w.tolist())
            M_si.extend(M.tolist())
            V_si.extend(V.tolist())

        # Convert results to the specified output units
        xs_out = np.array(xs)
        w_out = np.array(w_si)
        M_out = np.array(M_si)
        V_out = np.array(V_si)
        
        # Calculate stress: sigma = M*y/I
        # y_top is h/2, y_bot is -h/2
        sigma_top_si = M_out * (self.h / 2.0) / self.I
        sigma_bot_si = M_out * (-self.h / 2.0) / self.I

        if self.unit_system == 'N-mm-MPa':
            x_out = xs_out * 1000.0          # m -> mm
            w_out = w_out * 1000.0           # m -> mm
            M_out = M_out * 1000.0           # N·m -> N·mm
            V_out = V_out                    # N -> N
            sigma_top_out = sigma_top_si / 1e6 # Pa -> MPa
            sigma_bot_out = sigma_bot_si / 1e6 # Pa -> MPa
        elif self.unit_system == 'kN-m-kPa':
            x_out = xs_out                   # m -> m
            w_out = w_out * 1000.0           # m -> mm (displaying in mm for readability)
            M_out = M_out / 1e3              # N·m -> kN·m
            V_out = V_out / 1e3              # N -> kN
            sigma_top_out = sigma_top_si / 1e3 # Pa -> kPa
            sigma_bot_out = sigma_bot_si / 1e3 # Pa -> kPa
        
        return x_out, w_out, M_out, V_out, sigma_top_out, sigma_bot_out

# ============ CLI Wrapping / Plotting ============

def _plot_with_annotations(x_data, y_data, title, y_label, color, segment_boundaries, x_unit):
    """A helper function to plot a single field with max/min annotations."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color=color)
    plt.title(title)
    plt.xlabel(f"x ({x_unit})")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.axhline(0, color='k', linewidth=0.5)
    for xb in segment_boundaries:
        plt.axvline(xb * (1000.0 if x_unit == 'mm' else 1.0), color='k', linestyle='--', linewidth=2)

    # Annotate max and min values
    if y_data.size > 0:
        max_val = np.max(y_data)
        min_val = np.min(y_data)
        x_at_max = x_data[np.argmax(y_data)]
        x_at_min = x_data[np.argmin(y_data)]

        # Adjust text position to avoid overlap
        y_max_text_offset = max_val * 0.1 if max_val != 0 else 0.1
        y_min_text_offset = min_val * 0.1 if min_val != 0 else -0.1
        
        plt.annotate(
            f"Max: {max_val:.2f} {y_label.split(' ')[-1]}",
            xy=(x_at_max, max_val),
            xytext=(x_at_max, max_val + y_max_text_offset),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center'
        )
        
        plt.annotate(
            f"Min: {min_val:.2f} {y_label.split(' ')[-1]}",
            xy=(x_at_min, min_val),
            xytext=(x_at_min, min_val + y_min_text_offset),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center'
        )
    plt.tight_layout()

def print_menu():
    print("\n***MAIN MENU***")
    print("0. Exit Program")
    print("1. Define Beam Properties")
    print("2. Add Supports")
    print("3. Add Loads")
    print("4. Run Analysis (solve)")
    print("5. Show Reactions")
    print("6. Plot Results (Deflection, Moment, Shear, Stress)")
    print("7. Start New Beam")

def define_multi_span_beam(unit_system):
    spans = []
    total_elements = 0
    MAX_ELEMENTS = 2000
    while True:
        print(f"\nDefining span #{len(spans)+1}:")
        
        E = float(input(f"  Young's modulus E ({'MPa' if unit_system == 'N-mm-MPa' else 'kPa'}): "))
        props = get_section_props(unit_system)
        
        length_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
        length = float(input(f"  Length ({length_unit}): "))
        
        while True:
            ne = int(input(f"  Number of elements for this span (remaining allowed: {MAX_ELEMENTS - total_elements}): "))
            if 20 <= ne <= (MAX_ELEMENTS - total_elements):
                break
            else:
                print(f"  ✗ Please input a number between 20 and {MAX_ELEMENTS - total_elements}.")
        
        spans.append({
            'length': length,
            'E_val': E,
            'I_val': props['I'],
            'h_val': props['h'],
            'num_elements': ne
        })
        total_elements += ne
        if total_elements >= MAX_ELEMENTS:
            print(f"  ✓ Maximum total elements ({MAX_ELEMENTS}) reached.")
            break
        more = input("Add another span? (y/n): ").strip().lower()
        if more != "y":
            break
    return MultiSpanBeam(spans, unit_system)

def get_section_props(unit_system):
    """Prompts the user to select and define a cross-section."""
    props = {}
    print("\nSelect cross-section type:")
    print("  1. I-shaped")
    print("  2. Box-shaped")
    print("  3. Arbitrary")
    
    while True:
        try:
            choice = input("Enter 1, 2, or 3: ").strip()
            
            if choice == "1":
                unit = "mm" if unit_system == "N-mm-MPa" else "m"
                print(f"  You selected I-shaped. All dimensions in {unit}.")
                b_f = float(input("    Flange width (b_f): "))
                t_f = float(input("    Flange thickness (t_f): "))
                h_w = float(input("    Web height (h_w): "))
                t_w = float(input("    Web thickness (t_w): "))
                
                I = (b_f * (2*t_f + h_w)**3) / 12 - ((b_f - t_w) * h_w**3) / 12
                h = 2*t_f + h_w
                props = {'I': I, 'h': h}
                break
            
            elif choice == "2":
                unit = "mm" if unit_system == "N-mm-MPa" else "m"
                print(f"  You selected Box-shaped. All dimensions in {unit}.")
                b_o = float(input("    Outer width (b_o): "))
                h_o = float(input("    Outer height (h_o): "))
                b_i = float(input("    Inner width (b_i): "))
                h_i = float(input("    Inner height (h_i): "))
                
                if b_i >= b_o or h_i >= h_o:
                    print("  ✗ Error: Inner dimensions must be smaller than outer dimensions.")
                    continue
                
                I = (b_o * h_o**3) / 12 - (b_i * h_i**3) / 12
                h = h_o
                props = {'I': I, 'h': h}
                break
            
            elif choice == "3":
                I_unit = "mm^4" if unit_system == "N-mm-MPa" else "m^4"
                h_unit = "mm" if unit_system == "N-mm-MPa" else "m"
                print("  You selected Arbitrary. Input I and h directly.")
                I = float(input(f"    Second moment I ({I_unit}): "))
                h = float(input(f"    Height h ({h_unit}): "))
                props = {'I': I, 'h': h}
                break
            
            else:
                print("  ✗ Invalid selection. Please enter 1, 2, or 3.")
        except ValueError:
            print("  ✗ Invalid input. Please enter a valid number for the dimensions.")
    return props

def add_support_cli(beam):
    pos_unit = "mm" if beam.unit_system == 'N-mm-MPa' else "m"
    while True:
        try:
            pos = float(input(f"Support position ({pos_unit}): "))
            if pos < 0 or pos > beam.L * (1000.0 if pos_unit == "mm" else 1.0):
                print(f"  ✗ Please enter a number between 0 and {beam.L * (1000.0 if pos_unit == 'mm' else 1.0):.3f} ({pos_unit} beam length).")
                continue
            
            # Confirmation step
            while True:
                confirm = input(f"Confirm support position at {pos:.3f} {pos_unit}? (y/n): ").strip().lower()
                if confirm == 'y':
                    break  # Break inner loop to continue with support type selection
                elif confirm == 'n':
                    print("  Re-entering support position...")
                    break  # Break inner loop to re-enter position
                else:
                    print("  ✗ Invalid input. Please enter 'y' or 'n'.")
            if confirm == 'y':
                break  # Break outer loop to continue with support type selection
            
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
    print(f"  ✓ Support added at x = {pos:.3f} {pos_unit} ({typ})")

def add_load_cli(beam):
    print("Select load type:")
    print("  1. Point load")
    print("  2. Uniformly distributed load (UDL)")
    print("  3. Point moment")
    print("  4. Linearly varying distributed load (LVDL)")
    type_map = {"1": "point", "2": "udl", "3": "moment", "4": "lvdl"}
    while True:
        sel = input("Type (1/2/3/4): ").strip()
        if sel in type_map:
            t = type_map[sel]
            break
        else:
            print("  ✗ Invalid selection. Please enter 1, 2, 3, or 4.")

    pos_unit = "mm" if beam.unit_system == 'N-mm-MPa' else "m"
    force_unit = "N" if beam.unit_system == 'N-mm-MPa' else "kN"
    udl_unit = "N/mm" if beam.unit_system == 'N-mm-MPa' else "kN/m"
    lvdl_unit = "N/mm" if beam.unit_system == 'N-mm-MPa' else "kN/m"
    moment_unit = "N·mm" if beam.unit_system == 'N-mm-MPa' else "kN·m"

    if t == "point":
        while True:
            try:
                x = float(input(f"  Position x ({pos_unit}): "))
                if x < 0 or x > beam.L * (1000 if pos_unit == "mm" else 1.0):
                    print(f"  ✗ Please enter a number between 0 and {beam.L * (1000 if pos_unit == 'mm' else 1.0):.3f} ({pos_unit} beam length).")
                    continue
                
                P = float(input(f"  Load P ({force_unit}) [downward negative]: "))
                
                # Confirmation step for both values
                while True:
                    confirm = input(f"Confirm point load of {P:.3f} {force_unit} at {x:.3f} {pos_unit}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    elif confirm == 'n':
                        print("  Re-entering point load details...")
                        break
                    else:
                        print("  ✗ Invalid input. Please enter 'y' or 'n'.")
                if confirm == 'y':
                    break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for position and load.")
        beam.add_point_load(x, P)
        print(f"  ✓ Point load added at x = {x:.3f} {pos_unit}: {P:.3f} {force_unit}")
    elif t == "udl":
        while True:
            try:
                x1 = float(input(f"  Start x1 ({pos_unit}): "))
                x2 = float(input(f"  End   x2 ({pos_unit}): "))
                if x1 < 0 or x2 > beam.L * (1000 if pos_unit == "mm" else 1.0) or x1 >= x2:
                    print(f"  ✗ Please enter valid start/end positions within [0, {beam.L * (1000 if pos_unit == 'mm' else 1.0):.3f}] and x1 < x2.")
                    continue
                
                w  = float(input(f"  Intensity w ({udl_unit}) [downward negative]: "))
                
                # Confirmation step
                while True:
                    confirm = input(f"Confirm UDL of {w:.3f} {udl_unit} on [{x1:.3f}, {x2:.3f}] {pos_unit}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    elif confirm == 'n':
                        print("  Re-entering UDL details...")
                        break
                    else:
                        print("  ✗ Invalid input. Please enter 'y' or 'n'.")
                if confirm == 'y':
                    break
            except ValueError:
                print("  ✗ Invalid input. Please enter valid numbers for start/end positions.")
        beam.add_udl(x1, x2, w)
        print(f"  ✓ UDL added on [{x1:.3f}, {x2:.3f}] {pos_unit}: {w:.3f} {udl_unit}")
    elif t == "moment":
        while True:
            try:
                x = float(input(f"  Position x ({pos_unit}): "))
                if x < 0 or x > beam.L * (1000 if pos_unit == "mm" else 1.0):
                    print(f"  ✗ Please enter a number between 0 and {beam.L * (1000 if pos_unit == 'mm' else 1.0):.3f} ({pos_unit} beam length).")
                    continue
                
                M = float(input(f"  Moment M ({moment_unit}) [CCW positive]: "))
                
                # Confirmation step
                while True:
                    confirm = input(f"Confirm moment of {M:.3f} {moment_unit} at {x:.3f} {pos_unit}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    elif confirm == 'n':
                        print("  Re-entering moment details...")
                        break
                    else:
                        print("  ✗ Invalid input. Please enter 'y' or 'n'.")
                if confirm == 'y':
                    break
            except ValueError:
                print("  ✗ Invalid input. Please enter a valid number for position and moment.")
        beam.add_point_moment(x, M)
        print(f"  ✓ Point moment added at x = {x:.3f} {pos_unit}: {M:.3f} {moment_unit}")
    elif t == "lvdl":
        while True:
            try:
                x1 = float(input(f"  Start x1 ({pos_unit}): "))
                x2 = float(input(f"  End x2 ({pos_unit}): "))
                if x1 < 0 or x2 > beam.L * (1000 if pos_unit == "mm" else 1.0) or x1 >= x2:
                    print(f"  ✗ Please enter valid start/end positions within [0, {beam.L * (1000 if pos_unit == 'mm' else 1.0):.3f}] and x1 < x2.")
                    continue
                
                w1 = float(input(f"  Start intensity w1 ({lvdl_unit}) [downward negative]: "))
                w2 = float(input(f"  End intensity w2 ({lvdl_unit}) [downward negative]: "))
                
                # Confirmation step
                while True:
                    confirm = input(f"Confirm LVDL on [{x1:.3f}, {x2:.3f}] {pos_unit} from {w1:.3f} to {w2:.3f} {lvdl_unit}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    elif confirm == 'n':
                        print("  Re-entering LVDL details...")
                        break
                    else:
                        print("  ✗ Invalid input. Please enter 'y' or 'n'.")
                if confirm == 'y':
                    break
            except ValueError:
                print("  ✗ Invalid input. Please enter valid numbers for positions and intensities.")
        beam.add_lvdl(x1, x2, w1, w2)
        print(f"  ✓ LVDL added on [{x1:.3f}, {x2:.3f}] {pos_unit} from {w1:.3f} to {w2:.3f} {lvdl_unit}")
    else:
        print("  ✗ Unknown load type.")

def plot_results(x, w, M, V, sigma_top, sigma_bot, beam):
    """
    Generates and displays plots for deflection, bending moment, shear force,
    and stress distribution along the beam, with max/min values labeled.
    """
    
    # Detect multi-span and segment boundaries
    segment_boundaries = []
    if hasattr(beam, "spans"):
        cum_length = 0.0
        for span in beam.spans[:-1]:
            if beam.unit_system == 'N-mm-MPa':
                cum_length += span['length'] / 1000.0
            else:
                cum_length += span['length']
            segment_boundaries.append(cum_length)
    
    # Units based on user selection
    x_unit = "mm" if beam.unit_system == 'N-mm-MPa' else "m"
    w_unit = "mm"
    M_unit = "N·mm" if beam.unit_system == 'N-mm-MPa' else "kN·m"
    V_unit = "N" if beam.unit_system == 'N-mm-MPa' else "kN"
    sigma_unit = "MPa" if beam.unit_system == 'N-mm-MPa' else "kPa"
    
    # Deflection Plot
    _plot_with_annotations(x, w, "Deflected Shape", f"Deflection ({w_unit})", 'b', segment_boundaries, x_unit)
    
    # Bending Moment Plot
    _plot_with_annotations(x, M, "Bending Moment Diagram (BMD)", f"Moment ({M_unit})", 'r', segment_boundaries, x_unit)
    
    # Shear Force Plot
    _plot_with_annotations(x, V, "Shear Force Diagram (SFD)", f"Shear ({V_unit})", 'g', segment_boundaries, x_unit)
    
    # Stress Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, sigma_top, 'r-', label="Top Fiber")
    plt.plot(x, sigma_bot, 'b--', label="Bottom Fiber")
    plt.title("Stress Distribution")
    plt.xlabel(f"x ({x_unit})")
    plt.ylabel(r"Stress ($\sigma$, {sigma_unit})")
    plt.grid(True)
    plt.axhline(0, color='k', linewidth=0.5)
    for xb in segment_boundaries:
        plt.axvline(xb * (1000.0 if x_unit == 'mm' else 1.0), color='k', linestyle='--', linewidth=2)
    
    # Annotate max/min stress values
    all_stress = np.concatenate([sigma_top, sigma_bot])
    if all_stress.size > 0:
        max_stress = np.max(all_stress)
        min_stress = np.min(all_stress)
        x_at_max = x[np.argmax(sigma_top)] if np.max(sigma_top) > np.max(sigma_bot) else x[np.argmax(sigma_bot)]
        x_at_min = x[np.argmin(sigma_bot)] if np.min(sigma_bot) < np.min(sigma_top) else x[np.argmin(sigma_top)]
        
        plt.annotate(
            f"Max: {max_stress:.2f} {sigma_unit}",
            xy=(x_at_max, max_stress),
            xytext=(x_at_max, max_stress + max_stress * 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center'
        )
        plt.annotate(
            f"Min: {min_stress:.2f} {sigma_unit}",
            xy=(x_at_min, min_stress),
            xytext=(x_at_min, min_stress + min_stress * 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center'
        )
        
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("\n*** STRUCTURAL BEAM ANALYSER (Euler–Bernoulli FEM, CLI) ***")
    print("This program allows you to analyze a beam and see its deflection, moment, and shear.")
    print("First, please select a consistent unit system to use for all inputs.\n")
    
    beam = None
    solved = False
    d = R_full = reactions = None
    x = w = M = V = sigma_top = sigma_bot = None
    
    # Prompt for unit system once at the start
    while True:
        print("Select unit system:")
        print("  1. N, mm, MPa")
        print("  2. kN, m, kPa")
        try:
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                unit_system = "N-mm-MPa"
                print("  ✓ Unit system set to N, mm, MPa.")
                break
            elif choice == "2":
                unit_system = "kN-m-kPa"
                print("  ✓ Unit system set to kN, m, kPa.")
                break
            else:
                print("  ✗ Invalid selection.")
        except Exception as e:
            print("  ✗ Error:", e)

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
                    beam = define_multi_span_beam(unit_system)
                    print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                else:
                    length_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
                    E_unit = "MPa" if unit_system == 'N-mm-MPa' else "kPa"
                    L = float(input(f"Beam length L ({length_unit}): "))
                    E = float(input(f"Young's modulus E ({E_unit}): "))
                    props = get_section_props(unit_system)
                    I = props['I']
                    h = props['h']
                    
                    while True:
                        ne = int(input("Number of elements (recommend 20 — 2,000): "))
                        if 20 <= ne <= 2000:
                            break
                        else:
                            print("  ✗ Please input a number between 20 to 2000.")
                    beam = Beam(L, E, I, h, num_elements=ne, unit_system=unit_system)
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
                            break
                        elif again == "n":
                            break
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
                            break
                        elif again == "n":
                            break
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
                x, w, M, V, sigma_top, sigma_bot = beam.field_along_beam(d, npts_per_el=40)
                elapsed_sec = time.time() - start
                solved = True
                print(f"  ✓ Analysis complete. (Solved in {elapsed_sec:.1f} seconds)")
            except Exception as e:
                print("  ✗ Solve failed:", e)

        elif choice == "6":
            if not solved:
                print("  ✗ Run analysis first (option 4).")
                continue
            plot_results(x, w, M, V, sigma_top, sigma_bot, beam)
            print("  ✓ Results plotted.")

        elif choice == "5":
            if not solved:
                print("  ✗ Run analysis first (option 4).")
                continue
            # Show only supported DOFs as reactions
            print("\n=== Reaction Forces/Moments at Supports ===")
            force_unit = "N" if unit_system == 'N-mm-MPa' else "kN"
            moment_unit = "N·mm" if unit_system == 'N-mm-MPa' else "kN·m"
            
            for (node, dof) in sorted(set(beam.supports)):
                rxn = reactions.get((dof, node), 0.0)
                x_pos_m = beam.x_nodes[node]
                
                if unit_system == 'N-mm-MPa':
                    x_pos_out = x_pos_m * 1000.0
                    rxn_out = rxn
                    moment_out = rxn
                else:
                    x_pos_out = x_pos_m
                    rxn_out = rxn / 1e3
                    moment_out = rxn / 1e3
                    
                if dof == 0:
                    print(f" x = {x_pos_out:.3f} {('mm' if unit_system == 'N-mm-MPa' else 'm')} : V = {rxn_out:.3f} {force_unit}")
                else:
                    print(f" x = {x_pos_out:.3f} {('mm' if unit_system == 'N-mm-MPa' else 'm')} : M = {moment_out:.3f} {moment_unit}")
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
                        try:
                            while True:
                                multi = input("Multi-span beam? (y/n): ").strip().lower()
                                if multi in ("y", "n"):
                                    break
                                else:
                                    print("  ✗ Please type 'y' or 'n'.")
                            if multi == "y":
                                beam = define_multi_span_beam(unit_system)
                                print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                            else:
                                length_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
                                E_unit = "MPa" if unit_system == 'N-mm-MPa' else "kPa"
                                I_unit = "mm^4" if unit_system == 'N-mm-MPa' else "m^4"
                                h_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
                                L = float(input(f"Beam length L ({length_unit}): "))
                                E = float(input(f"Young's modulus E ({E_unit}): "))
                                I = float(input(f"Second moment I ({I_unit}): "))
                                h = float(input(f"Beam height h ({h_unit}): "))
                                while True:
                                    ne = int(input("Number of elements (recommend 20 — 2,000): "))
                                    if 20 <= ne <= 2000:
                                        break
                                    else:
                                        print("  ✗ Please input a number between 20 to 2000.")
                                beam = Beam(L, E, I, h, num_elements=ne, unit_system=unit_system)
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
                try:
                    while True:
                        multi = input("Multi-span beam? (y/n): ").strip().lower()
                        if multi in ("y", "n"):
                            break
                        else:
                            print("  ✗ Please type 'y' or 'n'.")
                    if multi == "y":
                        beam = define_multi_span_beam(unit_system)
                        print("  ✓ Multi-span beam created with {} elements.".format(beam.ne))
                    else:
                        length_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
                        E_unit = "MPa" if unit_system == 'N-mm-MPa' else "kPa"
                        I_unit = "mm^4" if unit_system == 'N-mm-MPa' else "m^4"
                        h_unit = "mm" if unit_system == 'N-mm-MPa' else "m"
                        L = float(input(f"Beam length L ({length_unit}): "))
                        E = float(input(f"Young's modulus E ({E_unit}): "))
                        I = float(input(f"Second moment I ({I_unit}): "))
                        h = float(input(f"Beam height h ({h_unit}): "))
                        while True:
                            ne = int(input("Number of elements (recommend 20 — 2,000): "))
                            if 20 <= ne <= 2000:
                                break
                            else:
                                print("  ✗ Please input a number between 20 to 2000.")
                        beam = Beam(L, E, I, h, num_elements=ne, unit_system=unit_system)
                        print("  ✓ Beam created with {} elements.".format(ne))
                    solved = False
                except Exception as e:
                    print("  ✗ Error defining beam:", e)

if __name__ == "__main__":

    main()
