# Qiskit Local Setup 

## 1. Check the existing Python

Open Terminal and go to the project:

```bash
cd ~/.../quantum-computing-lab
```

Check Python:

```bash
python3 --version
```

Also check the architecture:

```bash
uname -m
```

---

## 2. Install Homebrew Python 3.12

Check whether Homebrew is installed:

```bash
brew --version
```

If Homebrew is available, install Python 3.12:

```bash
brew install python@3.12
```

Verify:

```bash
python3.12 --version
```

Expected:

```text
Python 3.12.x
```

In our setup, Python 3.12.14 was used.

---

## 3. Create the Python virtual environment

Go to the project:

```bash
cd ~/.../quantum-computing-lab
```

Create a new virtual environment using Python 3.12:

```bash
python3.12 -m venv .venv
```

---

## 4. Activate the virtual environment

Run:

```bash
source .venv/bin/activate
```

The Terminal prompt should show that the virtual environment is active, for example:

```text
(.venv) ➜ quantum-computing-lab
```

The exact appearance of the prompt can vary depending on the shell configuration.

---

## 5. Verify Python inside `.venv`

Run:

```bash
python --version
```

Expected:

```text
Python 3.12.x
```

Check which Python is being used:

```bash
which python
```

Expected path:

```text
/.../quantum-computing-lab/.venv/bin/python
```

This confirms that commands are using the project's virtual environment rather than the system Python.

---

## 6. Upgrade pip

With `.venv` activated:

```bash
python -m pip install --upgrade pip
```

Using:

```bash
python -m pip
```

is preferred because it ensures that pip belongs to the currently active Python environment.

---

## 7. Install Qiskit

Install the main Qiskit package:

```bash
python -m pip install qiskit
```

For local quantum-circuit simulation, install Qiskit Aer:

```bash
python -m pip install qiskit-aer
```

Install the visualization and notebook packages used in this project:

```bash
python -m pip install matplotlib pylatexenc jupyter seaborn
```

The complete installation can also be performed with:

```bash
python -m pip install qiskit qiskit-aer matplotlib pylatexenc jupyter seaborn
```

---

## 8. Verify Qiskit

Check the installed Qiskit version:

```bash
python -c "import qiskit; print('Qiskit:', qiskit.__version__)"
```

A successful installation should print the installed Qiskit version.

Check Qiskit Aer:

```bash
python -c "from qiskit_aer import AerSimulator; print('Aer: OK')"
```

Expected:

```text
Aer: OK
```

---

## 9. Open the project in VS Code

From the project directory:

```bash
code .
```

If the `code` command is not available, open VS Code normally and use:

**File → Open Folder → `quantum-computing-lab`**

---

## 10. Install VS Code extensions

In VS Code, open Extensions with:

```text
⌘ + Shift + X
```

Install:

1. **Python** — Microsoft
2. **Jupyter** — Microsoft

No separate Qiskit VS Code extension is required for this setup.

---

## 11. Select the Qiskit virtual environment in VS Code

In VS Code:

```text
⌘ + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Select the interpreter corresponding to:

```text
quantum-computing-lab/.venv/bin/python
```

It may appear as:

```text
Python 3.12.x ('.venv': venv)
```

### Why this matters

VS Code can otherwise use a different Python installation. Qiskit may be installed correctly in `.venv`, but VS Code could report:

```text
ModuleNotFoundError: No module named 'qiskit'
```

if it is using another interpreter.

---

# 12. Set up a Jupyter Notebook in VS Code

Create:

```text
quantum_basics.ipynb
```

Open it in VS Code.

At the top-right, select the kernel/interpreter corresponding to:

```text
Python 3.12.x (.venv)
```

It should display the circuit, bloch sphere and q-shere graphically.

---

# 13. Add `.venv` to `.gitignore`

Create `.gitignore`:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

Do **not** commit `.venv` to Git.

The virtual environment should be recreated on another machine rather than uploaded to GitHub.

---

# 14. Reactivating the environment later

Every time you open a new Terminal session and want to work on this project:

```bash
cd ~/.../quantum-computing-lab
source .venv/bin/activate
```

Then verify:

```bash
python --version
```

and:

```bash
which python
```

When finished, you can leave the environment with:

```bash
deactivate
```