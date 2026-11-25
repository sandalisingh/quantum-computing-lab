## Boolean-to-Quantum Gate Translation Table



| Boolean Operation | Algebraic Form | Quantum Gate(s) | Initial Target State ($t$) | Circuit Implementation Steps |
| :--- | :--- | :--- | :--- | :--- |
| **COPY (Fan-out)** | $y = a$ | **CNOT** | $\|0\rangle$ | Apply `CNOT(a, t)`. Target becomes $\|a\rangle$. |
| **NOT** | $y = \neg a$ | **X** | $\|a\rangle$ (in-place) | Apply `X(a)`. Qubit flips in-place without needing an ancilla. |
| **XOR** | $y = a \oplus b$ | **CNOT** ($\times 2$) | $\|0\rangle$ | Apply `CNOT(a, t)`, then `CNOT(b, t)`. *(Or `CNOT(a, b)` to overwrite $b$ in-place).* |
| **XNOR** | $y = \neg(a \oplus b)$ | **CNOT** ($\times 2$) + **X** | $\|0\rangle$ | Compute XOR onto $t$, then apply `X(t)`. *(Or initialize $t = \|1\rangle$).* |
| **AND (2-bit)** | $y = a \land b$ | **CCX** (Toffoli) | $\|0\rangle$ | Apply `CCX(a, b, t)` with $a, b$ as controls. |
| **NAND (2-bit)** | $y = \neg(a \land b)$ | **CCX** + **X** | $\|0\rangle$ | Apply `CCX(a, b, t)`, then `X(t)`. *(Or initialize $t = \|1\rangle$).* |
| **OR (2-bit)** | $y = a \lor b$ | **X** + **CCX** | $\|0\rangle$ | Use De Morgan's Law ($\neg(\neg a \land \neg b)$): Apply `X(a)`, `X(b)`, then `CCX(a, b, t)`, then `X(t)`, and uncompute `X(a)`, `X(b)`. |
| **NOR (2-bit)** | $y = \neg(a \lor b)$ | **X** + **CCX** | $\|0\rangle$ | Apply `X(a)`, `X(b)`, then `CCX(a, b, t)`, then uncompute `X(a)`, `X(b)`. |
| **AND ($n$-bit)** | $y = a_1 \land a_2 \dots \land a_n$ | **MCX** ($\text{C}^n\text{X}$) | $\|0\rangle$ | Apply MCX controlled on all $n$ input qubits targeting $t$. |
| **OR ($n$-bit)** | $y = a_1 \lor a_2 \dots \lor a_n$ | **X** + **MCX** | $\|0\rangle$ | Apply `X` to all $n$ inputs, apply MCX onto $t$, apply `X(t)`, then uncompute `X` on all $n$ inputs. |
| **MUX (If-Then-Else)** | $y = (s \land a) \lor (\neg s \land b)$ | **CSWAP** (Fredkin) | In-place | Apply `CSWAP(s, a, b)`. If $s=1$, $a$ and $b$ swap; otherwise they stay unchanged. |
| **Half Adder** | $\text{Sum} = a \oplus b$<br>$\text{Carry} = a \land b$ | **CCX** + **CNOT** | $\|0\rangle$ (for Carry) | Apply `CCX(a, b, t)` for Carry, then `CNOT(a, b)` to store Sum in $b$. |
