# tqp-py

Python bindings for the TQP simulator, enabling algorithm development in Python.

## Installation

Requires `maturin`.

```bash
pip install maturin
maturin develop
```

## Usage

```python
import tqp
import math

# Initialize state
state = tqp.PyTQPState(2, 1, 1)

# Apply Gates
# ... (See examples/vqe_demo.py)

# Calculate Expectation Value
z_val = state.expval_z(0)
print(f"<Z_0>: {z_val}")
```

## Examples

Check the `examples/` directory for:

- `vqe_demo.py`: Variational Quantum Eigensolver for H2.
- `qaoa_demo.py`: QAOA for MaxCut problem.
