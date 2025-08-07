use generator::{done, Generator, Gn, Scope};

pub(crate) struct WeightedRangeComboGenerator {
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

    pub fn generate(&self) -> Generator<'static, (), Vec<usize>> {
        let items = self.items.clone();
        let bridge_cost = self.bridge_cost;
        let max_removal_weight = self.bridge_nodes * self.max_node_weight;
        let total_available_weight: usize = self.items.iter().map(|item| item.1).sum();

        Gn::new_scoped(move |mut scope| {
            fn backtrack<'scope>(
                scope: &mut Scope<'scope, 'static, (), Vec<usize>>,
                items: &[(usize, usize)],
                index: usize,
                current_combination: Vec<(usize, usize)>,
                current_weight: usize,
                target_weight: usize,
                total_available_weight: usize,
            ) {
                if current_weight >= target_weight {
                    scope.yield_(current_combination.iter().map(|item| item.0).collect());
                    return;
                }
                if index == items.len() {
                    return;
                }
                if current_weight + total_available_weight < target_weight {
                    return;
                }
                // Include current item
                backtrack(
                    scope,
                    items,
                    index + 1,
                    {
                        let mut new_comb = current_combination.clone();
                        new_comb.push(items[index]);
                        new_comb
                    },
                    current_weight + items[index].1,
                    target_weight,
                    total_available_weight - items[index].1,
                );
                // Exclude current item
                backtrack(
                    scope,
                    items,
                    index + 1,
                    current_combination,
                    current_weight,
                    target_weight,
                    total_available_weight,
                );
            }

            for target_weight in bridge_cost..=max_removal_weight {
                backtrack(
                    &mut scope,
                    &items,
                    0,
                    vec![],
                    0,
                    target_weight,
                    total_available_weight,
                );
            }
            done!()
        })
    }
}
