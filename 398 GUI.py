from tkinter import *
import re

user_input = Tk()
user_input.title('Beam Calculator')

# Configure grid columns
for i in range(12):
    user_input.columnconfigure(i, weight=1)

# Conversion factors
length_factors = {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1e3}
E_factors = {"GPa": 1, "MPa": 1e-3, "KPa": 1e-6}
I_factors = {"mm^4": 1, "cm^4": 1e4, "m^4": 1e12}
load_factors = {"N": 1, "kN": 1e3, "MN": 1e6}
load_factors_per_length = {"N/m": 1, "kN/m": 1e3, "MN/m": 1e6}
position_factors = {"mm": 1e-3, "cm": 1e-2, "m": 1}

# Dropdown options
length_units = ['mm', 'cm', 'm']
E_units = ['GPa', 'MPa', 'KPa']
I_units = ['mm^4', 'cm^4', 'm^4']
udl_units = ['N/m', 'kN/m', 'MN/m']
P_units = ['N', 'kN', 'MN']
position_units = ['mm', 'cm', 'm']

# Track spans
spans = []

# Status label
status_label = Label(user_input, text="", fg="red")
status_label.grid(column=0, row=0, columnspan=12, sticky="w")

# --- Validation functions ---
def validate_positive_num(num_input, label):
    pattern = r'^\d*\.?\d*([eE]\+?\d+)?$'
    if num_input == "" or re.match(pattern, num_input):
        label.config(text="", fg="black")
        run_button.config(state="normal")
        status_label.config(text="")
        return True
    else:
        label.config(text="Invalid input (must be positive)", fg="red")
        run_button.config(state="disabled")
        return False

def validate_any_num(num_input, label):
    pattern = r'^[-+]?\d*\.?\d*([eE][-+]?\d+)?$'
    if num_input == "" or re.match(pattern, num_input):
        label.config(text="", fg="black")
        run_button.config(state="normal")
        status_label.config(text="")
        return True
    else:
        label.config(text="Invalid input (must be a number)", fg="red")
        run_button.config(state="disabled")
        return False

# --- Span add/remove functions ---
span_selector_var = StringVar(user_input, "1")
span_selector_menu = None

def add_span():
    span_index = len(spans)
    row = span_index * 6 + 2  # allocate 6 rows per span (labels + entries + loads)
    
    # Span specifications labels
    Label(user_input, text=f"Span {span_index+1} Specifications").grid(column=0, row=row, sticky="w")
    labels_text = ["Length", "Units", "E", "Units", "I", "Units"]
    for i, text in enumerate(labels_text):
        Label(user_input, text=text).grid(column=i, row=row+1)
    
    # Validation labels
    v_labels = [Label(user_input) for _ in range(9)]  # 6 loads + 3 for main fields
    for i, v in enumerate(v_labels):
        v.grid(column=0, row=row+2+i, sticky="w")
    
    # Entry widgets
    length_entry = Entry(user_input, validate='key')
    length_entry['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[0])), '%P')
    length_entry.grid(column=0, row=row+2)
    length_units_var = StringVar(user_input, "m")
    OptionMenu(user_input, length_units_var, *length_units).grid(column=1, row=row+2)

    E_entry = Entry(user_input, validate='key')
    E_entry['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[1])), '%P')
    E_entry.grid(column=2, row=row+2)
    E_units_var = StringVar(user_input, "GPa")
    OptionMenu(user_input, E_units_var, *E_units).grid(column=3, row=row+2)

    I_entry = Entry(user_input, validate='key')
    I_entry['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[2])), '%P')
    I_entry.grid(column=4, row=row+2)
    I_units_var = StringVar(user_input, "mm^4")
    OptionMenu(user_input, I_units_var, *I_units).grid(column=5, row=row+2)

    # --- Loads Section ---
    Label(user_input, text="Loads").grid(column=0, row=row+3, sticky="w")
    
    # UDL
    Label(user_input, text="UDL (pos down, neg up)").grid(column=0, row=row+4, sticky="w")
    udl_entry = Entry(user_input, validate='key')
    udl_entry['validatecommand'] = (user_input.register(lambda P: validate_any_num(P, v_labels[3])), '%P')
    udl_entry.grid(column=0, row=row+5)
    udl_units_var = StringVar(user_input, "N/m")
    OptionMenu(user_input, udl_units_var, *udl_units).grid(column=1, row=row+5, sticky="w")
    
    # Linear Distributed Load
    Label(user_input, text="Linear Load (start to end)").grid(column=2, row=row+4, sticky="w")
    LDL_entry = Entry(user_input, validate='key')
    LDL_entry['validatecommand'] = (user_input.register(lambda P: validate_any_num(P, v_labels[4])), '%P')
    LDL_entry.grid(column=2, row=row+5)
    LDL_units_var = StringVar(user_input, "N/m")
    OptionMenu(user_input, LDL_units_var, *udl_units).grid(column=3, row=row+5, sticky="w")
    
    # LDL Start/End locations
    LDL_start_loc = Entry(user_input, validate='key')
    LDL_start_loc['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[5])), '%P')
    LDL_start_loc.grid(column=4, row=row+5)
    LDL_start_units_var = StringVar(user_input, "m")
    OptionMenu(user_input, LDL_start_units_var, *position_units).grid(column=5, row=row+5, sticky="w")
    
    LDL_end_loc = Entry(user_input, validate='key')
    LDL_end_loc['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[6])), '%P')
    LDL_end_loc.grid(column=6, row=row+5)
    LDL_end_units_var = StringVar(user_input, "m")
    OptionMenu(user_input, LDL_end_units_var, *position_units).grid(column=7, row=row+5, sticky="w")
    
    # Point Load
    Label(user_input, text="Point Load (pos down, neg up)").grid(column=8, row=row+4, sticky="w")
    P_entry = Entry(user_input, validate='key')
    P_entry['validatecommand'] = (user_input.register(lambda P: validate_any_num(P, v_labels[7])), '%P')
    P_entry.grid(column=8, row=row+5)
    P_units_var = StringVar(user_input, "N")
    OptionMenu(user_input, P_units_var, *P_units).grid(column=9, row=row+5, sticky="w")
    
    P_loc_entry = Entry(user_input, validate='key')
    P_loc_entry['validatecommand'] = (user_input.register(lambda P: validate_positive_num(P, v_labels[8])), '%P')
    P_loc_entry.grid(column=10, row=row+5)
    P_loc_units_var = StringVar(user_input, "m")
    OptionMenu(user_input, P_loc_units_var, *position_units).grid(column=11, row=row+5, sticky="w")
    
    spans.append({
        "label": None,
        "length": length_entry, "E": E_entry, "I": I_entry,
        "UDL": udl_entry, "LDL": LDL_entry, "LDL_start": LDL_start_loc, "LDL_end": LDL_end_loc,
        "P": P_entry, "P_loc": P_loc_entry,
        "v_labels": v_labels,
        "length_units": length_units_var, "E_units": E_units_var, "I_units": I_units_var,
        "UDL_units": udl_units_var, "LDL_units": LDL_units_var, "LDL_start_units": LDL_start_units_var, "LDL_end_units": LDL_end_units_var,
        "P_units": P_units_var, "P_loc_units": P_loc_units_var
    })
    
    update_span_selector()
    update_buttons()

def remove_span():
    if len(spans) <= 1:
        return
    try:
        idx = int(span_selector_var.get()) - 1
    except ValueError:
        return
    if 0 <= idx < len(spans):
        span = spans.pop(idx)
        # Destroy all widgets
        span["label"].destroy()
        keys = ["length","E","I","UDL","LDL","LDL_start","LDL_end","P","P_loc"]
        for key in keys:
            span[key].destroy()
        for v in span["v_labels"]:
            v.destroy()
        # Destroy unit menus
        for var in ["length_units","E_units","I_units","UDL_units","LDL_units",
                    "LDL_start_units","LDL_end_units","P_units","P_loc_units"]:
            menu = span[var]._tk.globalgetvar(span[var]._name)
        # Re-grid remaining spans
        for i, s in enumerate(spans):
            row = i*7+2
            s["label"].grid(column=0, row=row, sticky="w")
            s["label"].config(text=f"Span {i+1}")
            s["length"].grid(column=1, row=row+2)
            s["E"].grid(column=3, row=row+2)
            s["I"].grid(column=5, row=row+2)
            s["UDL"].grid(column=0, row=row+4)
            s["LDL"].grid(column=1, row=row+4)
            s["LDL_start"].grid(column=2, row=row+4)
            s["LDL_end"].grid(column=3, row=row+4)
            s["P"].grid(column=4, row=row+4)
            s["P_loc"].grid(column=5, row=row+4)
        update_span_selector()
        update_buttons()

def update_span_selector():
    global span_selector_menu
    if span_selector_menu:
        span_selector_menu.destroy()
    span_numbers = [str(i+1) for i in range(len(spans))]
    span_selector_var.set(span_numbers[0])
    span_selector_menu = OptionMenu(user_input, span_selector_var, *span_numbers)
    span_selector_menu.grid(row=len(spans)*7+3, column=5)

def get_user_input():
    outputs = []
    for i, span in enumerate(spans, start=1):
        errors = []
        try:
            L_str = span["length"].get()
            E_str = span["E"].get()
            I_str = span["I"].get()
            if not L_str: errors.append("Length missing")
            if not E_str: errors.append("E missing")
            if not I_str: errors.append("Moment of Inertia missing")

            # Optional loads default to 0
            udl_str = span["UDL"].get() or "0"
            LDL_str = span["LDL"].get() or "0"
            LDL_start_str = span["LDL_start"].get() or "0"
            LDL_end_str = span["LDL_end"].get() or "0"
            P_str = span["P"].get() or "0"
            P_loc_str = span["P_loc"].get() or "0"

            if errors: raise ValueError(", ".join(errors))

            # Convert to float
            L = float(L_str)
            E = float(E_str)
            I = float(I_str)
            udl = float(udl_str)
            LDL = float(LDL_str)
            LDL_start = float(LDL_start_str)
            LDL_end = float(LDL_end_str)
            P = float(P_str)
            P_loc = float(P_loc_str)

            if P_loc < 0 or P_loc > L:
                raise ValueError(f"Point Load location ({P_loc}) must be between 0 and {L}")

        except ValueError as ve:
            status_label.config(text=f"⚠️ Span {i} error: {ve}")
            return None

        # Units
        L_unit = span["length_units"].get()
        E_unit = span["E_units"].get()
        I_unit = span["I_units"].get()
        udl_unit = span["UDL_units"].get()
        LDL_unit = span["LDL_units"].get()
        LDL_start_unit = span["LDL_start_units"].get()
        LDL_end_unit = span["LDL_end_units"].get()
        P_unit = span["P_units"].get()
        P_loc_unit = span["P_loc_units"].get()

        # Convert to base units
        L_m = L * length_factors[L_unit]
        E_GPa = E * E_factors[E_unit]
        I_mm4 = I * I_factors[I_unit]
        udl_N = udl * load_factors_per_length[udl_unit] * L_m
        LDL_N = LDL * load_factors_per_length[LDL_unit] * L_m
        LDL_start_m = LDL_start * position_factors[LDL_start_unit]
        LDL_end_m = LDL_end * position_factors[LDL_end_unit]
        P_N = P * load_factors[P_unit]
        P_loc_m = P_loc * position_factors[P_loc_unit]

        outputs.append([L_m, E_GPa, I_mm4, udl_N, LDL_N, LDL_start_m, LDL_end_m, P_N, P_loc_m])

    status_label.config(text="✅ Inputs valid.", fg="green")
    return outputs

def run_and_store():
    data = get_user_input()
    if data:
        print("All spans (converted):", data)

def update_buttons():
    row = len(spans)*7 + 3
    run_button.grid(row=row, column=2)
    add_button.grid(row=row, column=3)
    remove_button.grid(row=row, column=4)
    span_selector_menu.grid(row=row, column=5)
    exit_button.grid(row=row, column=6)
    remove_button.config(state="normal" if len(spans) > 1 else "disabled")
    status_label.grid(column=0, row=row+1, columnspan=12, sticky="w")


run_button = Button(user_input, text="Run analysis", command=lambda: print("Run logic here"))
add_button = Button(user_input, text="Add Span", command=add_span)
remove_button = Button(user_input, text="Remove Span", command=lambda: print("Remove logic here"))
exit_button = Button(user_input, text="Exit", command=user_input.destroy)

# Initial span
add_span()
user_input.mainloop()
