# Setup

## Python setup
First create a python virtual environment and install ryd-numerov
```bash
uv venv .venv --python 3.13
uv pip install ryd-numerov
```

## Julia setup
Then install the required julia packages, and link against the above create python environment:
Run julia via `julia --project=.` and then

```julia
using Pkg
Pkg.instantiate()
ENV["PYTHON"] = "FULL/PATH/TO/.venv/bin/python"
Pkg.build("PyCall")
```

## Run the test script
Now you can run the `julia_ryd_numerov.jl` script via
```bash
julia --project=. julia_ryd_numerov.jl
```

Alternatively, you can also use the julia jupyter notebook `julia_ryd_numerov.ipynb` by selecting the julia kernel in jupyter notebook.
