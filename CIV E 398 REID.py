from tkinter import *
import re

user_input = Tk()
user_input.title('Bernoulli Beam Calculator')

# Configure grid columns
for i in range(11):
    user_input.columnconfigure(i, weight=1)

# Column labels
Label(user_input, text='Length of Beam').grid(column=1, row=1)
Label(user_input, text='Units').grid(column=2, row=1)
Label(user_input, text='Young’s Modulus').grid(column=3, row=1)
Label(user_input, text='Units').grid(column=4, row=1)
Label(user_input, text='Moment of Inertia').grid(column=5, row=1)
Label(user_input, text='Units').grid(column=6, row=1)
Label(user_input, text='Uniform Load w (N/m)').grid(column=7, row=1)
Label(user_input, text='Point Load P (N)').grid(column=8, row=1)
Label(user_input, text='P Location (m)').grid(column=9, row=1)

# Conversion factors
length_factors = {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1e3}
E_factors = {"GPa": 1e9, "MPa": 1e6, "KPa": 1e3}  # Pa
I_factors = {"mm^4": 1e-12, "cm^4": 1e-8, "m^4": 1}  # m^4

# Unit options
length_units = ['mm', 'cm', 'm', 'km']
E_units = ['GPa', 'MPa', 'KPa']
I_units = ['mm^4', 'cm^4', 'm^4']

# Keep track of spans
spans = []

# Status label at top
status_label = Label(user_input, text="", fg="red")
status_label.grid(column=0, row=0, columnspan=11, sticky="w")

# Validation function
def validate_num(num_input, label, name="", span_number=0):
    """Only check if input is numeric"""
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
    if re.match(pattern, num_input) or num_input == "":
        label.config(text="", fg="black")
        status_label.config(text="")
        run_button.config(state="normal")
        return True
    else:
        label.config(text="Invalid", fg="red")
        status_label.config(text=f"⚠️ Invalid input in Span {span_number}: {name}")
        run_button.config(state="disabled")
        return False

# --- Functions to add/remove spans ---
span_selector_var = StringVar(user_input, "1")
span_selector_menu = None

def add_span():
    row = len(spans) + 2
    span_number = len(spans) + 1

    # Span label
    span_label = Label(user_input, text=f"Span {span_number}")
    span_label.grid(column=0, row=row)

    # Validation labels
    v1 = Label(user_input); v1.grid(column=1, row=row+1)
    v2 = Label(user_input); v2.grid(column=3, row=row+1)
    v3 = Label(user_input); v3.grid(column=5, row=row+1)
    v4 = Label(user_input); v4.grid(column=7, row=row+1)
    v5 = Label(user_input); v5.grid(column=8, row=row+1)
    v6 = Label(user_input); v6.grid(column=9, row=row+1)

    # Entries with updated validate commands
    length_entry = Entry(user_input, validate='key')
    length_entry['validatecommand'] = (
        user_input.register(lambda P, lbl=v1: validate_num(P, lbl, "Length", span_number)), '%P')
    length_entry.grid(column=1, row=row)

    E_entry = Entry(user_input, validate='key')
    E_entry['validatecommand'] = (
        user_input.register(lambda P, lbl=v2: validate_num(P, lbl, "E", span_number)), '%P')
    E_entry.grid(column=3, row=row)

    I_entry = Entry(user_input, validate='key')
    I_entry['validatecommand'] = (
        user_input.register(lambda P, lbl=v3: validate_num(P, lbl, "I", span_number)), '%P')
    I_entry.grid(column=5, row=row)

    w_entry = Entry(user_input, validate='key')
    w_entry['validatecommand'] = (
        user_input.register(lambda P, lbl=v4: validate_num(P, lbl, "w", span_number)), '%P')
    w_entry.grid(column=7, row=row)

    P_entry = Entry(user_input, validate='key')
    P_entry['validatecommand'] = (
        user_input.register(lambda P, lbl=v5: validate_num(P, lbl, "P", span_number)), '%P')
    P_entry.grid(column=8, row=row)

    P_loc_entry = Entry(user_input, validate='key')
    P_loc_entry['validatecommand'] = (
    user_input.register(lambda P, lbl=v6: validate_num(P, lbl, "P Location", span_number)), '%P')
    P_loc_entry.grid(column=9, row=row)



    # Dropdowns
    length_units_var = StringVar(user_input, "m")
    E_units_var = StringVar(user_input, "GPa")
    I_units_var = StringVar(user_input, "mm^4")
    length_menu = OptionMenu(user_input, length_units_var, *length_units)
    E_menu = OptionMenu(user_input, E_units_var, *E_units)
    I_menu = OptionMenu(user_input, I_units_var, *I_units)
    length_menu.grid(column=2, row=row)
    E_menu.grid(column=4, row=row)
    I_menu.grid(column=6, row=row)

    # Add span to list
    spans.append({
        "label": span_label, "length": length_entry, "E": E_entry, "I": I_entry,
        "w": w_entry, "P": P_entry, "P_loc": P_loc_entry,
        "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6,
        "length_units": length_units_var, "E_units": E_units_var, "I_units": I_units_var,
        "menus": [length_menu, E_menu, I_menu]
    })

    update_buttons()
    update_span_selector()

def remove_span():
    if len(spans) <= 1:
        return
    try:
        idx = int(span_selector_var.get()) - 1
    except ValueError:
        return
    if 0 <= idx < len(spans):
        span = spans.pop(idx)
        for key in ["label", "length", "E", "I", "w", "P", "P_loc", "v1","v2","v3","v4","v5","v6"]:
            span[key].destroy()
        for menu in span["menus"]:
            menu.destroy()
        # Repack remaining spans
        for i, s in enumerate(spans, start=1):
            row = i + 1
            s["label"].grid(column=0, row=row)
            s["label"].config(text=f"Span {i}")
            s["length"].grid(column=1, row=row)
            s["E"].grid(column=3, row=row)
            s["I"].grid(column=5, row=row)
            s["w"].grid(column=7, row=row)
            s["P"].grid(column=8, row=row)
            s["P_loc"].grid(column=9, row=row)
            s["v1"].grid(column=1, row=row+1)
            s["v2"].grid(column=3, row=row+1)
            s["v3"].grid(column=5, row=row+1)
            s["v4"].grid(column=7, row=row+1)
            s["v5"].grid(column=8, row=row+1)
            s["v6"].grid(column=9, row=row+1)
            s["menus"][0].grid(column=2, row=row)
            s["menus"][1].grid(column=4, row=row)
            s["menus"][2].grid(column=6, row=row)
    update_buttons()
    update_span_selector()

def update_span_selector():
    global span_selector_menu
    if span_selector_menu:
        span_selector_menu.destroy()
    span_numbers = [str(i+1) for i in range(len(spans))]
    span_selector_var.set(span_numbers[0])
    span_selector_menu = OptionMenu(user_input, span_selector_var, *span_numbers)
    span_selector_menu.grid(row=len(spans)+3, column=5)

def get_user_input():
    outputs = []
    for i, span in enumerate(spans, start=1):
        try:
            L = float(span["length"].get())
            E = float(span["E"].get())
            I = float(span["I"].get())
            w = float(span["w"].get()) if span["w"].get() else 0
            P = float(span["P"].get()) if span["P"].get() else 0
            P_loc = float(span["P_loc"].get()) if span["P_loc"].get() else 0
        except ValueError:
            status_label.config(text=f"⚠️ Invalid number in Span {i}.")
            return None

        # Check that P location is valid
        if P_loc < 0 or P_loc > L:
            status_label.config(text=f"⚠️ Point load location P in Span {i} must be between 0 and {L}")
            return None

        # Convert units
        L_unit = span["length_units"].get()
        E_unit = span["E_units"].get()
        I_unit = span["I_units"].get()

        L_m = L * length_factors[L_unit]
        E_GPa = E * E_factors[E_unit]
        I_mm4 = I * I_factors[I_unit]

        outputs.append({
        "L": L_m,
        "E": E_GPa,
        "I": I_mm4,
        "w": w,
        "P": P,
        "P_loc": P_loc,
        "delta_max": None,  # placeholder if you will calculate later
        "M_max": None})


    status_label.config(text="✅ Inputs valid.", fg="green")
    return outputs


def run_and_store():
    data = get_user_input()
    if data:
        text = "Span Results:\n"
        for i, d in enumerate(data, start=1):
            text += f"Span {i}: δ_max={d['delta_max']:.6f} m, M_max={d['M_max']:.2f} N·m\n"
        results_label.config(text=text)

# Result display
results_label = Label(user_input, text="", justify=LEFT)
results_label.grid(column=0, row=100, columnspan=11, sticky="w")

# Buttons
run_button = Button(user_input, text="Run analysis", command=run_and_store)
add_button = Button(user_input, text="Add Span", command=add_span)
remove_button = Button(user_input, text="Remove Span", command=remove_span)
exit_button = Button(user_input, text="Exit", command=user_input.destroy)

def update_buttons():
    row = len(spans) + 3
    run_button.grid(row=row, column=2)
    add_button.grid(row=row, column=3)
    remove_button.grid(row=row, column=4)
    if span_selector_menu:
        span_selector_menu.grid(row=row, column=5)
    exit_button.grid(row=row, column=6)
    remove_button.config(state="normal" if len(spans) > 1 else "disabled")

# Add first span
add_span()

user_input.mainloop()
