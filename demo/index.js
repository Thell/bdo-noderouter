import init, { WasmNodeRouter } from "./pkg/nwsf_rust.js";

let router = null;

document.getElementById("solveBtn").addEventListener("click", async () => {
    const output = document.getElementById("output");
    output.textContent = "";

    if (!router) {
        output.textContent = "Error: Load a valid exploration.json first.";
        return;
    }

    const rawInput = document.getElementById("terminalPairs").value.trim();
    const pairs = rawInput
        .split(/\s+/)
        .map(pair => pair.split(",").map(x => parseInt(x.trim(), 10)))
        .filter(p => p.length === 2 && p.every(Number.isFinite));

    if (pairs.length === 0) {
        output.textContent = "Invalid input. Provide pairs like: 1,2 3,4";
        return;
    }

    try {
        const start = performance.now();
        const result = router.solveForTerminalPairs(pairs);
        const end = performance.now();

        output.textContent = `Result: [${result.join(", ")}]\nElapsed: ${(end - start).toFixed(2)}ms`;
    } catch (err) {
        output.textContent = `Error during solve: ${err}`;
    }
});

document.getElementById("fileInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const output = document.getElementById("output");
    output.textContent = "";

    if (!file) {
        output.textContent = "No file selected.";
        return;
    }

    try {
        const text = await file.text();
        const json = JSON.parse(text);

        await init(); // await WASM initialization
        router = new WasmNodeRouter(json);
        output.textContent = "exploration.json loaded successfully.";
    } catch (e) {
        output.textContent = `Failed to load/parse file: ${e.message || e}`;
    }
});
