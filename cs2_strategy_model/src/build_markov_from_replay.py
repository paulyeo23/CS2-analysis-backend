import os
import pandas as pd
from collections import defaultdict
import re


#player enters and exists zone quickly is called edging or jigging. This should help smoothen it out
def smooth_zones_for_player(player_df, min_ticks=32):
    player_df = player_df.sort_values("tick")
    zones = list(player_df["current_zone"])
    ticks = list(player_df["tick"])

    cleaned = []
    start_idx = 0

    # to remove i plan to go through zones one by one
    # if player stayed in zone for less time, they are jigging, scrap
    for i in range(1, len(zones)):
        if zones[i] != zones[i-1]:
            #print(f"{zones[i]} -> {zones[i-1]}")
            # figure out how long the player stayed in the old zone
            duration = i - start_idx
            zone_name = zones[start_idx]
            #print(f"{zone_name} duration: {duration}")
            # if too short duration treat as noise
            if duration < min_ticks:
                if len(cleaned) > 0:
                    cleaned_zone = cleaned[-1]
                    #print(f"which is cleaned zone: {cleaned_zone}")
                    for k in range(duration):
                        cleaned.append(cleaned_zone)
                else:
                    #print("Nothing cleaned")
                    for k in range(duration):
                        cleaned.append(zone_name)
            else:
                #print("this is good one")
                for k in range(duration):
                    cleaned.append(zone_name)

            start_idx = i

    #print(f"remaining ticks = {len(zones) - start_idx}")
    last_duration = len(zones) - start_idx
    last_zone = zones[start_idx]
    #print(last_zone)
    if last_duration < min_ticks and len(cleaned) > 0:
        cleaned_zone = cleaned[-1]
        for k in range(last_duration):
            cleaned.append(cleaned_zone)
    else:
        for k in range(last_duration):
            cleaned.append(last_zone)

    player_df["clean_zone"] = cleaned
    #print(player_df.head())
    return player_df

#clean all players
def smooth_all_players(df, min_ticks=32):
    df = df.sort_values(["steamid", "tick"])
    final_list = []
    for sid, group in df.groupby("steamid"):
        cleaned_group = smooth_zones_for_player(group, min_ticks)
        final_list.append(cleaned_group)
    new_df = pd.concat(final_list)
    return new_df


# single file transition matrix CONVERTED to multiple files
def build_transitions(df):
    transitions = defaultdict(lambda: defaultdict(int))

    for sid, group in df.groupby("steamid"):
        zones = list(group["clean_zone"])
        #print(f"zones: {zones}")
        for i in range(len(zones)-1):
            curr = zones[i]
            nxt = zones[i+1]
            #print(f"moving {curr} -> {nxt}")
            if curr != nxt:
                #print("is this a new place")
                transitions[curr][nxt] += 1
    #refactor for multiple replays
    #retdf = convert_trans_to_prob(transitions)
    #return retdf
    return transitions
    #i have to merge before converting to probs

#refactor to be reused
def convert_trans_to_prob(trans):
    rows = []
    for src in trans:
        total = sum(trans[src].values())
        # print(total)
        for dst, count in trans[src].items():
            prob = count / total
            rows.append([src, dst, count, prob])
            # print(f"{src}:{dst} has {count} and i calc pronb as {prob}")
    return pd.DataFrame(rows, columns=["from", "to", "count", "probability"])

def merge_transitions(master, new_trans):
    for src in new_trans:
        for dst in new_trans[src]:
            master[src][dst] += new_trans[src][dst]


# Function to convert the CSV data into a nested JSON structure
def trans_prob_to_json(trans,theroot="t_spawn"):
    trans_tree = defaultdict(list)
    print(trans)
    for i, row in trans.iterrows():
        trans_tree[row["from"]].append({"name": row["to"], "probability": row["probability"]})

    #doesnt work for maps interconnected. All nodes are roots.
    #all_children = set(trans["to"])
    #print(f"here are children: {all_children}")
    #all_parents = set(trans["from"])
    #print(f"here are parents: {all_parents}")
    #roots = list(all_parents - all_children)
    #print(f"roots is parents that are not children: {roots}")
    roots = [theroot]
    print(f"the root is {theroot}")
    if (theroot.casefold() == "all".casefold()):
            roots = list(trans_tree.keys())
    result = []
    for root in roots:
        level1_node = {"name": root, "children": []}
        has_children = len(trans_tree.get(root, [])) > 0
        print(f"root {root} has {has_children} children {trans_tree.get(root, [])}")
        #add children and then just their children, no more
        for child1 in trans_tree.get(root, []):
            #print(f"do i get the child for this node: {child1}")
            level2_node = {
                "name": child1["name"],
                "probability": child1["probability"],
                "children": []
            }
            for child2 in trans_tree.get(child1["name"], []):
                level3_node = {
                    "name": child2["name"],
                    "probability": child2["probability"]
                    # no more children
                }
                level2_node["children"].append(level3_node)
            level1_node["children"].append(level2_node)
        result.append(level1_node)

    return result

#single replay markoc
def build_markov(file_path, out_prefix, min_ticks=32):
    # try to read csv first
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        # otherwise try excel
        df = pd.read_excel(file_path)

    print("Loaded rows:", len(df))

    print("Smoothing zones...")
    df = smooth_all_players(df, min_ticks)

    print("Building transition matrix...")
    trans = build_transitions(df)
    trans_df = convert_trans_to_prob(trans)

    full_path = out_prefix + "_transitions.csv"
    trans_df.to_csv(full_path, index=False)

    simple_df = trans_df[["from", "to"]]
    simple_path = out_prefix + "_transitions_simple.csv"
    simple_df.to_csv(simple_path, index=False)

    print("Saved files:")
    print(" ", full_path)
    print(" ", simple_path)

    return trans_df

#to process multiple replays, i need to buidl coounts and then merge them
#convert this merged counts to probs
#code for multiple replay proessing usnig folders
def build_markov_from_folder(folder_path, output_prefix, min_ticks=32):
    total_trans = defaultdict(lambda: defaultdict(int))
    print("looking into folder:", folder_path)
    #all files in here are part of the same map
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv") or fname.endswith(".xlsx"):
            path = os.path.join(folder_path, fname)
            print("Processing:", path)
            if fname.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            #avoid jigging for each file
            df = smooth_all_players(df, min_ticks=min_ticks)
            #print(f"looked at {min_ticks} interval in the replay")
            # Transition counts from this replay
            replay_transitions = build_transitions(df)
            merge_transitions(total_trans, replay_transitions)

    # Convert final counts -> probabilities
    final_df = convert_trans_to_prob(total_trans)

    # Save full matrix
    full_csv = output_prefix + "_mul_transitions.csv"
    print(f"full path is {full_csv}")
    final_df.to_csv(full_csv, index=False)

    # Save simple neighbors (MarkovMap legacy)
    simple_csv = output_prefix + "_mul_transitions_simple.csv"
    final_df[["from", "to"]].to_csv(simple_csv, index=False)
    print(f"  Saved: {full_csv} and {simple_csv}")

    return final_df

def build_markov_for_all_maps(folder_path, output_dir, min_ticks=32):
    delim_start = "-"
    delim_end = "_ticks"
    ret = {}
    print("Scanning folder for replay tick files:", folder_path)
    #all map files are here
    map_files = defaultdict(list)

    for fname in os.listdir(folder_path):
        if fname.endswith(".csv") or fname.endswith(".xlsx"):
            if "_ticks" in fname:
                mapname = re.search(r"-([^-\n]+)_ticks", fname).group(1)
                #mapname = fname.split("_ticks")[0]
                print(f"mapname is {mapname}")
                map_files[mapname].append(fname)
    if len(map_files) == 0:
        print("No tick files found")
        return

    print("*******found maps:", list(map_files.keys()))
    #print(f"map_files is {map_files}")

    # per map processing
    for mapname, file_list in map_files.items():
        print(f"\nProcessing map: {mapname} having {len(file_list)} files")

        output_prefix = os.path.join(output_dir, mapname)

        temp_files = [os.path.join(folder_path, f) for f in file_list]

        total_trans = defaultdict(lambda: defaultdict(int))

        for filepath in temp_files:
            #print("  reading:", filepath)
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            df = smooth_all_players(df, min_ticks=min_ticks)
            replay_trans = build_transitions(df)
            merge_transitions(total_trans, replay_trans)
            #print(f"per replay transition matrix is: {replay_trans}")

        final_df = convert_trans_to_prob(total_trans)

        os.makedirs(output_dir, exist_ok=True)
        full_csv = output_prefix + "_transitions.csv"
        simple_csv = output_prefix + "_transitions_simple.csv"

        final_df.to_csv(full_csv, index=False)
        final_df[["from", "to"]].to_csv(simple_csv, index=False)

        print(f"  Saved: {full_csv} and {simple_csv}")
        if mapname not in ret:
            ret[mapname] = None
        ret[mapname] = final_df

    print("\nAll maps processed",ret)
    return ret

def filter_same_zone(df, current_col="current_zone", next_col="next_zone"):
    new_df = df[df[current_col] != df[next_col]].copy()
    print("Filtered rows:", len(df) - len(new_df))
    print("Remaining rows:", len(new_df))
    return new_df

def build_tests_markov_for_all_maps(folder_path, output_dir, min_ticks=32):
    ret = {}
    print("Scanning folder for test files:", folder_path)
    #all map files are here
    test_files = defaultdict(list)

    for fname in os.listdir(folder_path):
        if fname.endswith(".csv") or fname.endswith(".xlsx"):
            if "_ticks" in fname:
                mapname = re.search(r"-([^-\n]+)_ticks", fname).group(1)
                #mapname = fname.split("_ticks")[0]
                print(f"mapname is {mapname}")
                test_files[mapname].append(fname)
    if len(test_files) == 0:
        print("No test files found")
        return

    print("*******found test for maps:", list(test_files.keys()))
    #print(f"map_files is {test_files}")

    # per map processing
    for mapname, file_list in test_files.items():
        print(f"\nProcessing map: {mapname} having {len(file_list)} files")

        output_prefix = os.path.join(output_dir, mapname)
        temp_files = [os.path.join(folder_path, f) for f in file_list]
        dfs_to_combine = []

        for filepath in temp_files:
            #print("  reading:", filepath)
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            df = filter_same_zone(df)
            dfs_to_combine.append(df)
            #print(f"per replay transition matrix is: {replay_trans}")

        final_df = pd.concat(dfs_to_combine, ignore_index=True)

        if mapname not in ret:
            ret[mapname] = None
        ret[mapname] = final_df

    print("\nAll tests processed",ret)
    return ret



if __name__ == "__main__":
    # MIRAGE example
    '''build_markov(
        r"C:\gatech\ticks\aurora-vs-heroic-m1-mirage_ticks.xlsx",
        "../data/transition/mirage",
        min_ticks=32
    )'''

    '''build_markov_from_folder(
        folder_path=r"C:\gatech\ticks",
        output_prefix="../data/transition/mirage",
        min_ticks=32
    )'''

    build_markov_for_all_maps(folder_path=r"C:\gatech\ticks", output_dir="../data/transition/output",min_ticks=32)
