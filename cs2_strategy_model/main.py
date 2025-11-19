
from src.markov_model import MarkovMap
from src.build_markov_from_replay import build_markov_for_all_maps, build_tests_markov_for_all_maps, build_markov_from_folder, trans_prob_to_json
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from src.win_probability_model import WinProbabilityModel
import pandas as pd
from src.utils import load_map_transitions, plot_markov_graph
from collections import defaultdict

def main():
    output_prefix = "./data/transition/output"
    ticks_dir = "C:\\gatech\\ticks"
    test_markov_dir = "C:\\gatech\\ticks\\test"
    file_name_ts = "t_spwan_probabilities.json"
    file_name_all = "all_probabilities.json"
    min_ticks = 32
    transition_csv = "./data/transition/mirage_mul_transitions.csv"

    print("=== CS2 Strategy Model ===")
    print("\nBuilding transition matrix")
    #building the transition matrix from the replays fresh
    '''final_df = build_markov_from_folder(
        folder_path=ticks_dir,
        output_prefix=output_prefix,
        min_ticks=min_ticks
    )'''
    final_df_dict = build_markov_for_all_maps(folder_path=ticks_dir, output_dir=output_prefix,min_ticks=min_ticks)

    for mapname, final_df in final_df_dict.items():
        print("Processing map:", mapname)
        #print the json for Paul
        t_spwan_nodes = trans_prob_to_json(final_df,'t_spawn')
        all_nodes = trans_prob_to_json(final_df,'all')
        #print(f"\nThe JSON version of the final transition matrix is: \n{t_spwan_nodes}")
        #print(f"\nThe JSON version of the final transition matrix is: \n{all_nodes}")
        os.makedirs(output_prefix, exist_ok=True)
        os.makedirs(output_prefix + "/" + mapname, exist_ok=True)
        with open(output_prefix+"/"+mapname+"/"+file_name_ts, "w") as json_file:
            json.dump(t_spwan_nodes, json_file, indent=4)
        with open(output_prefix+"/"+mapname+"/"+file_name_all, "w") as json_file:
            json.dump(all_nodes, json_file, indent=4)

        # use the matrix to give player movement demo
        print("\nSimulating player movement...")
        #markov = MarkovMap(default_transitions)
        #markov = MarkovMap("./data/dust2_map_transitions.csv")
        markov = MarkovMap(transition_csv)
        path = markov.simulate_path("t_spawn", 12)
        print(" → ".join(path))
        #markov.plot_transition_graph()
        markov.plot_transition_graph_interactive(output_file=output_prefix+"/"+mapname+"/"+"markov_graph.html")
        #transitions = load_map_transitions("data/dust2_map_transitions.csv")

        #plot_markov_graph(transitions)

    ##testing
    test_folder = "test"
    final_test_df_dict = build_tests_markov_for_all_maps(folder_path=test_markov_dir, output_dir=output_prefix)
    for mapname, test_df in final_test_df_dict.items():
        print("Testing map:", mapname)
        print(f"transition file for {mapname} is {output_prefix}/{mapname}_transitions.csv")
        trans_matrix = markov.setup_testing(f"{output_prefix}/{mapname}_transitions.csv")
        result_df = markov.run_markov_on_test(trans_matrix, test_df, current_col="current_zone")
        accuracy = markov.compute_accuracy(result_df)
        print(f"Accuracy for predicting next move using markov chain for {mapname}: {accuracy}")
        labels = sorted(set(result_df["next_zone"]) | set(result_df["markov_pred"]))
        print("*****Labels are ", labels)
        cm = confusion_matrix(y_true=result_df["next_zone"], y_pred=result_df["markov_pred"], labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm,xticklabels=labels,yticklabels=labels,annot=False, cmap="plasma")
        plt.title(f"Markov Confusion Matrix for {mapname}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()
        plt.savefig(f"{output_prefix}/{mapname}/confusion_matrix.jpg", dpi=300, bbox_inches='tight')


'''    # --- Win Probability Model Demo ---
    print("\n[Neural Network] Training win probability model...")
    model = WinProbabilityModel()
    data = model.generate_dummy_data(3000)
    model.train(data, epochs=10)

    # Predict a sample state
    example_state = pd.DataFrame({
        'time_elapsed': [60],
        'team_money': [25000],
        'enemy_money': [20000],
        'team_alive': [4],
        'enemy_alive': [3],
        'bomb_planted': [1],
        'bomb_site_A': [1],
        'bomb_site_B': [0],
        'team_flashes': [2],
        'enemy_flashes': [1],
        'round_number': [12],
        'score_diff': [2],
    })

    model.model.save("win_prob_model.h5")

    win_prob = model.predict(example_state)
    print(f"\nPredicted Win Probability: {win_prob:.2%}")
    model.visualize_model()
'''
if __name__ == "__main__":
    main()
