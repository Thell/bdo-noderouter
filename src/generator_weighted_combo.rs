type StackState = (usize, Vec<(usize, usize)>, usize);

/// Generates combinations whose total weight is in [bridge_cost, bridge_nodes * max_node_weight].
///
/// NOTE: Changing of removal candidate generation alters outcomes...
/// - The difference is in the order of removal_set removals
/// - Results are all within +/- 1 on cost and #cc
/// - Example difference of +1 cost on workerman incident 265
/// - Example difference of -1 cost with +1 cc on workerman incident 410
///
/// This is why two passes are made in reverse directions.
/// Even if all removal_sets get processed it is not assured an optimal value
/// will be obtained because the order of application can improve the solution but
/// cause an optimal removal set to not be connected when applied.
pub struct WeightedRangeComboGenerator {
    items: Vec<(usize, usize)>, // (index, weight) pairs
    bridge_cost: usize,
    bridge_nodes: usize,
    max_node_weight: usize,
}

impl WeightedRangeComboGenerator {
    pub fn new(
        items: &[(usize, usize)],
        bridge_cost: usize,
        bridge_nodes: usize,
        max_node_weight: usize,
        combo_gen_direction: bool,
    ) -> Self {
        let mut items = items.to_vec();
        items.sort_by(|a, b| {
            if combo_gen_direction {
                b.1.cmp(&a.1) // Descending
            } else {
                a.1.cmp(&b.1) // Ascending
            }
        });
        WeightedRangeComboGenerator {
            items,
            bridge_cost,
            bridge_nodes,
            max_node_weight,
        }
    }

    pub fn generate(&self) -> WeightedComboIterator {
        let max_removal_weight = self.bridge_nodes * self.max_node_weight;
        WeightedComboIterator {
            items: self.items.clone(),
            state: BacktrackState::new(self.bridge_cost, max_removal_weight),
        }
    }
}

pub struct WeightedComboIterator {
    items: Vec<(usize, usize)>,
    state: BacktrackState,
}

struct BacktrackState {
    target_weight: usize,
    max_target_weight: usize,
    stack: Vec<StackState>, // (index, combination, current_weight)
    done: bool,
}

impl BacktrackState {
    fn new(bridge_cost: usize, max_target_weight: usize) -> Self {
        BacktrackState {
            target_weight: bridge_cost,
            max_target_weight,
            // Start empty with the first call: index 0, empty combination, weight 0
            stack: vec![(0, vec![], 0)],
            done: false,
        }
    }
}

impl Iterator for WeightedComboIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.state.done {
                return None;
            }

            // Check if we've exhausted all combinations for the current target_weight
            // and need to move to the next one.
            if self.state.stack.is_empty() {
                if self.state.target_weight < self.state.max_target_weight {
                    self.state.target_weight += 1;
                    // Reset the stack for the new target_weight
                    self.state.stack.push((0, vec![], 0));
                    continue;
                } else {
                    self.state.done = true;
                    return None;
                }
            }

            let (index, current_combination, current_weight) = self.state.stack.pop().unwrap();

            // Check for valid combinations
            if current_weight >= self.state.target_weight {
                // Yield the result and don't continue to explore this path
                // because we're looking for exact or greater matches.
                return Some(current_combination.iter().map(|item| item.0).collect());
            }

            // Pruning: if remaining items can't meet the target, we stop this path.
            let remaining_available_weight: usize =
                self.items[index..].iter().map(|item| item.1).sum();

            if current_weight + remaining_available_weight < self.state.target_weight {
                continue;
            }

            // Backtracking logic: Explore both "exclude" and "include" paths.
            if index < self.items.len() {
                // Path 1: Exclude the current item.
                // We have to reverse this from the generator as this gets pushed first
                // so it's explored after the "include" path.
                self.state
                    .stack
                    .push((index + 1, current_combination.clone(), current_weight));

                // Path 2: Include the current item.
                let mut new_comb = current_combination;
                new_comb.push(self.items[index]);
                let new_weight = current_weight + self.items[index].1;
                self.state.stack.push((index + 1, new_comb, new_weight));
            }
        }
    }
}
