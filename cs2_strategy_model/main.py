import argparse
from src.markov_model import MarkovMap
from src.build_markov_from_replay import build_markov_for_all_maps, build_tests_markov_for_all_maps, build_markov_from_folder, trans_prob_to_json
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from src.win_probability_model import WinProbabilityModel
from src.nn.nn_model_pipeline import run_nn_pipeline
import pandas as pd
from src.utils import load_map_transitions, plot_markov_graph
from collections import defaultdict

def main(ticks_dir):
    output_dir= "./output"
    output_markov = f"{output_dir}/markov"
    output_nn = f"{output_dir}/nn"
    #ticks_dir = r"C:\gatech\ticks"   ##get this as input for portabilituy
    test_ticks_dir = fr"{ticks_dir}\test" #same
    file_name_ts = "t_spwan_probabilities.json" #constant filename for saving
    file_name_all = "all_probabilities.json" #constant filename for saving
    min_ticks = 32
    os.makedirs(output_markov, exist_ok=True)
    os.makedirs(output_nn, exist_ok=True)
    print("=== CS2 Strategy Model ===")
    print("=== Markov pipeline ===")
    print("\nBuilding transition matrix")
    #building the transition matrix from the replays fresh
    '''final_df = build_markov_from_folder(
        folder_path=ticks_dir,
        output_prefix=output_prefix,
        min_ticks=min_ticks
    )'''


    final_df_dict = build_markov_for_all_maps(folder_path=ticks_dir, output_dir=output_markov,min_ticks=min_ticks)

    for mapname, final_df in final_df_dict.items():
        print("Processing map:", mapname)
        #print the json for Paul
        t_spwan_nodes = trans_prob_to_json(final_df,'t_spawn')
        all_nodes = trans_prob_to_json(final_df,'all')
        #print(f"\nThe JSON version of the final transition matrix is: \n{t_spwan_nodes}")
        #print(f"\nThe JSON version of the final transition matrix is: \n{all_nodes}")
        os.makedirs(output_markov, exist_ok=True)
        os.makedirs(output_markov + "/" + mapname, exist_ok=True)
        with open(output_markov+"/"+mapname+"/"+file_name_ts, "w") as json_file:
            json.dump(t_spwan_nodes, json_file, indent=4)
        with open(output_markov+"/"+mapname+"/"+file_name_all, "w") as json_file:
            json.dump(all_nodes, json_file, indent=4)

        # use the matrix to give player movement demo
        print("\nSimulating player movement...")
        #markov = MarkovMap(default_transitions)
        #markov = MarkovMap("./data/dust2_map_transitions.csv")
        markov = MarkovMap(f"{output_markov}/{mapname}_transitions.csv")
        path = markov.simulate_path("t_spawn", 12)
        print(" → ".join(path))
        #markov.plot_transition_graph()
        markov.plot_transition_graph_interactive(output_file=output_markov+"/"+mapname+"/"+"markov_graph.html")
        #transitions = load_map_transitions("data/dust2_map_transitions.csv")

        #plot_markov_graph(transitions)

    ##testing
    if not os.path.exists(test_ticks_dir):
        print(f"No tests present: {test_ticks_dir}")# graceful return
    else:
        final_test_df_dict = build_tests_markov_for_all_maps(folder_path=test_ticks_dir, output_dir=output_markov)
        for mapname, test_df in final_test_df_dict.items():
            print("Testing map:", mapname)
            print(f"transition file for {mapname} is {output_markov}/{mapname}_transitions.csv")
            trans_matrix = markov.setup_testing(f"{output_markov}/{mapname}_transitions.csv")
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
            plt.savefig(f"{output_markov}/{mapname}/markov_confusion_matrix.jpg", dpi=300, bbox_inches='tight')

    #markov done add the nn part here
    run_nn_pipeline(output_dir=output_nn,ticks_folder_path=ticks_dir)
'''    # --- Win Probability Model Demo ---
    #this was the concept for the progress. The project predection was for next mode and not winloss
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
    #main()
    parser = argparse.ArgumentParser(description="CS2 Strategy Model")

    parser.add_argument(
        "--ticks_dir",
        type=str,
        default=r"./data",
        help="Directory containing replay tick files. Add a test subfolder in here and drop in unseen replays. If not specified, local data will be used from ./data."
    )

    args = parser.parse_args()

    main(ticks_dir=args.ticks_dir)