import init, { WasmNodeRouter } from "./pkg/noderouter.js";

let router = null;

const testSets = {
    'no_super': '1826,1795 1827,1795 1828,1795 1829,1795 1879,1857 1880,1857 1881,1857 132,1 131,1 203,61 488,301 951,602 905,608 1815,1781 1816,1781 1820,1795 1821,1781 1830,1781 1823,1785 1886,1857 1882,1853 1884,1853 1885,1853 1889,1857 1893,1858 1894,1853 464,301 1813,1795 1814,1795 1890,1853 1891,1858 1892,1858 1895,1853 1896,1853 136,1 160,1 436,301 435,301 438,301 439,301 440,301 1513,1319 1527,1319 1531,1314 1530,1314 1529,1314 1538,1301 1536,1301 1537,1301 1807,1781 1808,1781 1809,1781 1904,1834 1899,1834 1883,1853 1897,1853 459,301 1534,1301 1535,1301 1565,1301 1887,1853 1888,1853 852,601 853,601 855,601 842,601 840,601 1912,602 1209,1101 1206,1141 1514,1314 1554,1301 1555,1380 1642,1623 1645,1623 1909,1843 188,61 908,608 1515,1314 1902,1843 1903,1843 1907,1843 1908,1843 1906,1843 1910,1843 176,61 175,61 135,61 172,61 171,61 183,1 184,61 144,1 443,301 1522,1319 1523,1319 1520,1319 1521,1319 455,301 476,301 480,301 1201,1141 1711,1691 1777,1750 1778,1750 1911,602 1208,1101 1204,1101 1561,1301 1562,1301 1558,1301 1556,1301 1710,1691 952,602 958,602 1203,1101 1516,1314 1636,1623 1637,1623 1771,1750 1772,1750 1517,1314 1716,1691 1769,1750 1770,1750 903,601 914,601 1502,1101 1501,1101 1504,1101 1505,1101 1508,1101 1507,1101 1687,1649 1913,602 910,602 913,601 912,601 1685,301 1686,301 1713,1691 1681,1649 1682,1649 1688,1649 1679,1649 1220,1141 1219,1141 1215,1141 1213,301 1212,301 1216,301 901,601 902,601 907,608 1683,302 1684,302',
    'super': '1826,1795 1827,1795 1828,1795 1829,1795 1879,1857 1880,1857 1881,1857 132,1 131,1 203,61 488,301 951,602 905,608 1815,1781 1816,1781 1820,1795 1821,1781 1830,1781 1823,1785 1886,1857 1882,1853 1884,1853 1885,1853 1889,1857 1893,1858 1894,1853 464,301 1813,1795 1814,1795 1890,1853 1891,1858 1892,1858 1895,1853 1896,1853 136,1 160,1 436,301 435,301 438,301 439,301 440,301 1513,1319 1527,1319 1531,1314 1530,1314 1529,1314 1538,1301 1536,1301 1537,1301 1807,1781 1808,1781 1809,1781 1904,1834 1899,1834 1883,1853 1897,1853 459,301 1534,1301 1535,1301 1565,1301 1887,1853 1888,1853 852,601 853,601 855,601 842,601 840,601 1912,602 1209,1101 1206,1141 1514,1314 1554,1301 1555,1380 1642,1623 1645,1623 1909,1843 188,61 908,608 1515,1314 1902,1843 1903,1843 1907,1843 1908,1843 1906,1843 1910,1843 176,61 175,61 135,61 172,61 171,61 183,1 184,61 144,1 443,301 1522,1319 1523,1319 1520,1319 1521,1319 455,301 476,301 480,301 1201,1141 1711,1691 1777,1750 1778,1750 1911,602 1208,1101 1204,1101 1561,1301 1562,1301 1558,1301 1556,1301 1710,1691 952,602 958,602 1203,1101 1516,1314 1636,1623 1637,1623 1771,1750 1772,1750 1517,1314 1716,1691 1769,1750 1770,1750 903,601 914,601 1502,1101 1501,1101 1504,1101 1505,1101 1508,1101 1507,1101 1687,1649 1913,602 910,602 913,601 912,601 1685,301 1686,301 1713,1691 1681,1649 1682,1649 1688,1649 1679,1649 1220,1141 1219,1141 1215,1141 1213,301 1212,301 1216,301 901,601 902,601 907,608 1683,302 1684,302 652,99999 324,99999 1146,99999 1132,99999 724,99999 373,99999'
};

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
        const opts = {
            max_removal_attempts: document.getElementById("maxRemovalAttempts").value,
            max_frontier_rings: document.getElementById("maxFrontierRings").value,
            ring_combo_cutoff: document.getElementById("ringComboCutoff").value,
        };
        console.log(opts);
        for (const [key, val] of Object.entries(opts)) {
            try {
                router.setOption(key, val);
            } catch (err) {
                output.innerHTML = `Error: Failed to set '${key}' — ${err}`;
                return;
            }
        }

        const start = performance.now();
        const [result, cost] = router.solveForTerminalPairs(pairs);
        const end = performance.now();

        output.innerHTML = `Elapsed: ${(end - start).toFixed(2)}ms, Cost: ${cost}<br>Result: [${result.join(", ")}]`;
    } catch (err) {
        output.innerHTML = `Error during solve: ${err}`;
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

        await init();
        router = new WasmNodeRouter(json);
        output.textContent = "exploration.json loaded successfully.";
        setActiveButton(null);
    } catch (e) {
        output.textContent = `Failed to load/parse file: ${e.message || e}`;
    }
});

function setActiveButton (activeButton) {
    document.querySelectorAll('.test-sets button').forEach(button => {
        button.classList.remove('active');
    });
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

document.querySelector(".test-sets").addEventListener("click", (event) => {
    const button = event.target.closest('button');
    if (button && button.dataset.testid) {
        const testId = button.dataset.testid;
        if (testSets[testId]) {
            document.getElementById("terminalPairs").value = testSets[testId];
            setActiveButton(button);
        }
    }
});

document.getElementById("terminalPairs").addEventListener("input", () => {
    setActiveButton(null);
});