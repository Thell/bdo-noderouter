# Usage notes:

- Requirements:
  - nightly for coroutines

## WASM

- Requirements:
  - wasm-pack for wasm
  - python to run the demo (or some other mini server)

- Build for wasm using
`wasm-pack build --release --target web --features wasm`

- To run the wasm demo:

```
> cp -Rf ./pkg ./demo/pkg
> cd demo
> python -m http.server 8080
```

Then load the clean exploration.json from `python/bdo_noderouter/data`.
Lastly input terminal,root pairs (as waypoint ids) and hit solve.

## Python

- Requirements:
  - python
  - uv
  - maturin for python

Setup project dependencies (you may need to setup a .venv if uv doesn't just do it for you):
`uv sync`

Build for Python using:
`maturin develop --profile release --features python --uv`

Run demo scripts...

For the standard Python implementation which is all python with a Rust IDTree
for connectivity testing...
`.venv/bin/python python/bdo_noderouter/pd_approximation.py`

For the full Rust implementation,
`.venv/bin/python python/bdo_noderouter/node_router_approximation.py`

Node: To see the terminal pairs and solution waypoints edit the
`pd_approximation.toml` logger level to 'INFO'.
