# api_nwst_types.py

from __future__ import annotations

from typing import TypedDict

# Primitive Types
type Mask = int  # bit represenation of any node indices
type NodeIndex = int
type RootIndex = int
type PotentialRootIndex = int
type SuperRootIndex = int
type SuperTerminalIndex = int
type TerminalIndex = int

type BlockMask = Mask
type Cost = int
type ConnectedComponentMappingKey = int
type CoveredEntity = NodeIndex
type CoverageRepresentative = NodeIndex
type SolutionMask = Mask

# Collection Types
type BlockKey = tuple[NodeIndex, ...]
type BlockedInteractionEdges = set[tuple[NodeIndex, NodeIndex]]
type ConnectedComponent_BlockKeys = dict[ConnectedComponentMappingKey, set[BlockKey]]
type CoverageBits = dict[RootIndex, Mask]
type CoverageRepresentatives = set[CoverageRepresentative]
type CoverageSets = dict[CoverageRepresentative, set[CoveredEntity]]
type SolutionSet = set[NodeIndex]
type TerminalsList = list[TerminalIndex]
type TerminalsSet = set[TerminalIndex]
type BlockCosts = dict[BlockKey, Cost]
type BlockResults = dict[BlockKey, SolutionMask]
type MaskedBlockCosts = dict[BlockMask, Cost]
type MaskedBlockResults = dict[BlockMask, BlockResults]
type MaskedBlockSolutionMasks = dict[BlockMask, SolutionMask]

type BlockTask = tuple[ConnectedComponentMappingKey, BlockKey, TerminalsList, NodeIndex]  # int is sr_index
type CompositeSolution = tuple[SolutionSet, Cost, BlockResults, BlockCosts]


class ConnectedComponentMappings(TypedDict):
    component: set[int]
    reachable: set[int]
    adj_map: dict[int, list[int]]
    nodes_list: list[int]
    node_index_map: dict[int, int]
    dimacs_id_map: dict[int, int]
    inv_dimacs_id_map: dict[int, int]
