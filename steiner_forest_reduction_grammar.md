# Steiner Forest Reduction Grammar (v1.0)

## 1. Symbol Glossary

### Nodes & Sets

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **Ⓣ** | Terminal Node | A node that *must* be connected to its partner. |
| **𝕥** | Super Terminal | A terminal whose demand is met by any node in **𝔹**. |
| **ⓢ** | Steiner Node | An optional node; no inherent weight/demand. |
| **Ⓡ** | Root Node | The root of an Arborescence (**𝓐**). |
| **𝕊** | Super Root | A "sink" for all potential root demands. |
| **𝔹** | Potential Roots | Set of nodes with edges to a Super Root. |
| **ⓑ** | Potential Root | An individual node member of **𝔹**. |
| **ⓤ, ⓥ, ⓦ** | Variable Nodes | Generic nodes often consumed in reductions. |
| **⦿** | Hyper-Node | A contracted component or cluster of nodes. |
| **○** | Neutral Placeholder | Used for general neighbors in abstract rules. |

### Connectivity & Topology

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **⇋** | Bi-directional Edge | Parallel edges or symmetric relationships. |
| **🡒** | Specific Path | A single, defined path between two nodes. |
| **↠** | Existence of Path | Represents any or all paths between nodes. |
| **|** | Boundary | Absolute boundary. |
| **⁝** | Universe Boundary | The "Tricolon" isolation cut. |
| **⋯** | Continuity | Indicates a path extends further into the graph. |
| **∞** | Infinite Set | The global "Universe" beyond a node. |

### Logic & Operators

| Symbol | Meaning | Role |
| :--- | :--- | :--- |
| **⭆** | Transform | The multi-operation reduction arrow. |
| **:** | Guard Delimiter | "Where / Such that" for local node properties. |
| **⇔** | Invariant | "If and only if" for global/attribute constraints. |
| **∈ / ∉** | Membership | Relative to an Arborescence **𝓐** or Set. |
| **\ / ⋃** | Set Ops | Exclusion (subtraction) and Union (addition). |
| **𝓐** | Arborescence Set | The cluster of nodes rooted at **Ⓡ**. |
| **𝐆** | Graph Set | Active nodes and edges. |
| **𝓢** | Solution Set | Nodes/Edges selected for the forest. |
| **𝐃** | Demand Set | Unsatisfied terminal pairs. |
| **𝔀( )** | Weight Function | The cost of a node or path. |

---

## 2. Grammar Structure
Rules follow a "Pattern-First" sequence:

**`|Pattern : Local Guards ⭆ Transform ⇔ Global/Attribute Guards`**

---

## 3. Current Valid Reductions

### A. Steiner Pruning (NTD1)
If a Steiner node is a leaf, it cannot satisfy or bridge a demand.
> **`|ⓢ ⇋ ○ ⇋ ⁝  ⭆  ○ ⇋ ⁝, 𝐆 \ ⓢ`**

### B. Terminal Consumption (TD1)
A terminal leaf must use its neighbor to reach the rest of the graph.
> **`|Ⓣ ⇋ ⓤ ⇋ ⁝  ⭆  Ⓣ ⇋ ⁝, 𝐆 \ ⓤ, 𝓢 ⋃ ⓤ`**

### C. Root-Neighbor Absorption
When a terminal leaf connects to a root it does not belong to, the root absorbs the terminal's logical demand.
> **`|Ⓣ ⇋ Ⓡ ⇋ ⁞ : Ⓣ ∉ 𝓐ᴿ ⭆ Ⓡ ⇋ ⁞, 𝐆 \ Ⓣ, 𝓐ᴿ ⋃ 𝓐ᵀ, 𝓢 ⋃ Ⓣ`**

### D. Adjacent Root Fusion
Two adjacent roots with zero weight are redundant and are fused.
*   **Bridge Case:** `⁝ ⇋ Ⓡᵏ ⇋ Ⓡʲ ⇋ ⁝ : 𝓐ᵏ ≠ ∅, 𝓐ʲ ≠ ∅  ⭆  𝓐ᵏ ⋃ 𝓐ʲ, 𝐆 \ Ⓡʲ ⇔ 𝔀(Ⓡᵏ)=0, 𝔀(Ⓡʲ)=0`
*   **Leaf Case:** `|Ⓡᵏ ⇋ Ⓡʲ ⇋ ⁝ : 𝓐ᵏ ≠ ∅, 𝓐ʲ ≠ ∅  ⭆  𝓐ᵏ ⋃ 𝓐ʲ, 𝐆 \ Ⓡʲ ⇔ 𝔀(Ⓡʲ)=0`

### E. Potential Root Entry (PRE)
A Super Terminal is satisfied by a Potential Root if the cost is zero or it is the dominant choice.
> **`|ⓑ ⇋ 𝕥 ⇋ ⁝ : 𝕥 ∈ 𝓐ˢ  ⭆  𝐆 \ 𝕥, 𝓐ˢ \ 𝕥, 𝓢 ⋃ 𝕥 ⇔ 𝔀(ⓑ)=0 ∨ ƒ(𝕥, ⓑ) ⭆ Dominance`**
