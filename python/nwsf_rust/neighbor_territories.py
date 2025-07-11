# neighbor_territories.py

import data_store as ds

# TODO: Retire this file after moving main somewhere.


def get_neighboring_territories(exploration_data: dict):
    """Processes exploration data to generate and return dict of `{territory: [neighbors]}`.
    Note: territory is included in neighbors

    If it is desired to omit the great ocean nodes then that should be done when creating
    the expxploration data.
    """
    territories = set()
    territory_pairs = set()
    for node_data in exploration_data.values():
        node_territory = node_data["territory_key"]
        territories.add(node_territory)
        for neighbor in node_data["link_list"]:
            neighbor_data = exploration_data.get(neighbor, None)
            if neighbor_data:
                neighbor_territory = neighbor_data["territory_key"]
                territories.add(neighbor_territory)
                territory_pairs.add((node_territory, neighbor_territory))

    results = {t: set() for t in territories}
    for t1, t2 in territory_pairs:
        results[t1].add(t2)
        results[t2].add(t1)
    results = {t: sorted(v) for t, v in results.items()}
    return results


def get_territory_root_sets(exploration_data: dict, territory_neighbors: dict):
    """Generate a dict of root nodes within a territory and its' neighbors."""
    territory_root_sets = {t: set() for t in territory_neighbors}

    for node_key, node_data in exploration_data.items():
        if not node_data["is_base_town"]:
            continue

        root_territory = node_data["territory_key"]
        if root_territory not in territory_neighbors:
            continue

        for n in territory_neighbors[root_territory]:
            territory_root_sets[n].add(node_key)

    territory_root_sets = {k: list(sorted(v)) for k, v in territory_root_sets.items()}
    return territory_root_sets


def generate_territory_root_sets(exploration_data: dict):
    territory_neighbors = get_neighboring_territories(exploration_data)
    territory_root_sets = get_territory_root_sets(exploration_data, territory_neighbors)
    return territory_root_sets


if __name__ == "__main__":
    import api_common as common_api

    config = ds.get_config("generate_testcase")
    exploration_data = common_api.get_clean_exploration_data(config)

    territory_neighbors = get_neighboring_territories(exploration_data)
    print(territory_neighbors)

    territory_root_sets = get_territory_root_sets(exploration_data, territory_neighbors)
    print(territory_root_sets)

    terminals_per_territory = {}
    for node_key, node_data in exploration_data.items():
        territory_key = node_data["territory_key"]
        terminals_per_territory[territory_key] = terminals_per_territory.get(territory_key, 0) + 1
    for territory_key, count in terminals_per_territory.items():
        print(f"{territory_key}: {count}")
    print(f"Total: {sum(terminals_per_territory.values())}")
