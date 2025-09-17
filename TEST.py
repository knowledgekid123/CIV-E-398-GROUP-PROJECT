from tkinter import *
import re

user_input = Tk()
user_input.title('Beam Calculator')

# Configure grid columns
for i in range(7):
    user_input.columnconfigure(i, weight=1)

# Column labels
Label(user_input, text='Length of Beam').grid(column=1, row=1)
Label(user_input, text='Select Units').grid(column=2, row=1)
Label(user_input, text='Young’s Modulus').grid(column=3, row=1)
Label(user_input, text='Select Units').grid(column=4, row=1)
Label(user_input, text='Moment of Inertia').grid(column=5, row=1)
Label(user_input, text='Select Units').grid(column=6, row=1)

# Conversion factors
length_factors = {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1e3}
E_factors = {"GPa": 1, "MPa": 1e-3, "KPa": 1e-6}
I_factors = {"mm^4": 1, "cm^4": 1e4, "m^4": 1e12}

# Dropdown unit options
length_units = ['mm', 'cm', 'm', 'km']
E_units = ['GPa', 'MPa', 'KPa']
I_units = ['mm^4', 'cm^4', 'm^4']

# Keep track of spans
spans = []

# Status label at bottom
status_label = Label(user_input, text="", fg="red")
status_label.grid(column=0, row=0, columnspan=7, sticky="w")

# Variables for E/I inputs
E_units_value = StringVar(user_input, "GPa")
I_units_value = StringVar(user_input, "mm^4")

# Validation function
def validate_num(num_input, label):
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
    if re.match(pattern, num_input) or num_input == "":
        label.config(text="", fg="black")
        run_button.config(state="normal")
        status_label.config(text="")
        return True
    else:
        label.config(text="", fg="red")
        run_button.config(state="disabled")
        status_label.config(text="⚠️ Invalid input detected. Please enter a valid number.")
        return False

# --- Functions to add/remove spans ---
span_selector_var = StringVar(user_input, "1")
span_selector_menu = None

def add_span():
    row = len(spans) + 2  # Row for this span

    # Span label
    span_label = Label(user_input, text=f"Span {len(spans)+1}")
    span_label.grid(column=0, row=row)

    # Validation labels
    v1 = Label(user_input); v1.grid(column=1, row=row+1)
    v2 = Label(user_input); v2.grid(column=3, row=row+1)
    v3 = Label(user_input); v3.grid(column=5, row=row+1)

    # Entries
    length_entry = Entry(user_input, validate='key')
    length_entry['validatecommand'] = (user_input.register(lambda P: validate_num(P, v1)), '%P')
    length_entry.grid(column=1, row=row)

    E_entry = Entry(user_input, validate='key')
    E_entry['validatecommand'] = (user_input.register(lambda P: validate_num(P, v2)), '%P')
    E_entry.grid(column=3, row=row)

    I_entry = Entry(user_input, validate='key')
    I_entry['validatecommand'] = (user_input.register(lambda P: validate_num(P, v3)), '%P')
    I_entry.grid(column=5, row=row)

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

    spans.append({
        "label": span_label,
        "length": length_entry,
        "E": E_entry,
        "I": I_entry,
        "v1": v1, "v2": v2, "v3": v3,
        "length_units": length_units_var,
        "E_units": E_units_var,
        "I_units": I_units_var,
        "menus": [length_menu, E_menu, I_menu]
    })

    update_span_selector()
    update_buttons()


def remove_span():
    if len(spans) <= 1:  # Keep at least one span
        return

    try:
        idx = int(span_selector_var.get()) - 1
    except ValueError:
        return

    if 0 <= idx < len(spans):
        span = spans.pop(idx)

        # Destroy all widgets for that span
        span["label"].destroy()
        span["length"].destroy()
        span["E"].destroy()
        span["I"].destroy()
        span["v1"].destroy()
        span["v2"].destroy()
        span["v3"].destroy()
        for menu in span["menus"]:
            menu.destroy()

        # Re-pack remaining spans to close the gap
        for i, s in enumerate(spans, start=1):
            s["label"].grid(column=0, row=i+1)
            s["label"].config(text=f"Span {i}")
            s["length"].grid(column=1, row=i+1)
            s["E"].grid(column=3, row=i+1)
            s["I"].grid(column=5, row=i+1)
            s["v1"].grid(column=1, row=i+2)
            s["v2"].grid(column=3, row=i+2)
            s["v3"].grid(column=5, row=i+2)
            s["menus"][0].grid(column=2, row=i+1)
            s["menus"][1].grid(column=4, row=i+1)
            s["menus"][2].grid(column=6, row=i+1)

        update_span_selector()
        update_buttons()


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
        except ValueError:
            status_label.config(text=f"⚠️ Invalid input in Span {i}.")
            return None

        L_unit = span["length_units"].get()
        E_unit = span["E_units"].get()
        I_unit = span["I_units"].get()

        # Convert
        L_m = L * length_factors[L_unit]
        E_GPa = E * E_factors[E_unit]
        I_mm4 = I * I_factors[I_unit]

        outputs.append([L_m, "m", E_GPa, "GPa", I_mm4, "mm^4"])

    status_label.config(text="✅ Inputs valid.", fg="green")
    return outputs


def run_and_store():
    data = get_user_input()
    if data:
        print("All spans (converted):", data)


def update_buttons():
    row = len(spans) + 3
    run_button.grid(row=row, column=2)
    add_button.grid(row=row, column=3)
    remove_button.grid(row=row, column=4)
    span_selector_menu.grid(row=row, column=5)
    exit_button.grid(row=row, column=6)

    # Disable remove button if only one span
    remove_button.config(state="normal" if len(spans) > 1 else "disabled")

    # Status label below buttons
    status_label.grid(column=0, row=row+1, columnspan=7, sticky="w")


# Buttons
run_button = Button(user_input, text="Run analysis", command=run_and_store)
add_button = Button(user_input, text="Add Span", command=add_span)
remove_button = Button(user_input, text="Remove Span", command=remove_span)
exit_button = Button(user_input, text="Exit", command=user_input.destroy)

# Add first span by default
add_span()

user_input.mainloop()
