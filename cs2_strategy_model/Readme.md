# CS2 Strategy Model

This project implements a **Markov Chain** and **Neural Network** pipeline to model and predict player movement in CS2 maps using replay tick data. It is designed for research and analysis of player strategies in CS2.

---

## Requirements

All Python dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Data

The project uses external replay tick data:

* You can pass the tick data directory as a command line argument:
```bash
python main.py --ticks_dir "C:\gatech\ticks"
```
* If no directory is provided, the project will use a default lightweight dataset from **./data** folder.

## Running the Project

Run the main script:
* Using default lightweight data:
```bash
python main.py
```
* Specify your tick data folder:
```bash
python main.py --ticks_dir "C:\gatech\ticks"
```
## Output Folder Structure
1. Markov Chain Output (./output/markov)
   
   This folder contains the results of the Markov pipeline:
* `<mapname>_transitions.csv` – Learned zone transition probabilities for each map.

Each map has its own subfolder `(./output/markov/<mapname>)` containing:

| File                          | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `t_spawn_probabilities.json`  | Transition probabilities starting from spawn nodes.               |
| `all_probabilities.json`      | Full transition probabilities for all zones.                      |
| `markov_confusion_matrix.jpg` | Confusion matrix from testing the Markov model on test tick data. |
| `markov_graph.html`           | Interactive visualization of the Markov chain.                    |

2. Neural Network Output (./output/nn)

   This folder contains the results of the Neural Network pipeline:

| File                                  | Description                                                    |
| ------------------------------------- | -------------------------------------------------------------- |
| `<mapname>_classification_report.txt` | Text report with classification metrics for each map.          |
| `<mapname>_confusion.png`             | Confusion matrix from testing the model on training data.      |
| `<mapname>_lstm.h5`                   | Learned LSTM model for the map.                                |
| `<mapname>_training.png`              | Loss and accuracy plots during training.                       |
| `<mapname>_newreplay_confusion.png`   | Confusion matrix when predicting on a new unseen replay graph. |

## Source Code Structure (`src/`)

The `src` folder contains the main modules and scripts for the project:

- **`build_markov_from_replay.py`** – Functions to parse tick data and build the Markov transition matrices for all maps.
- **`markov_model.py`** – Implementation of the `MarkovMap` class for simulating player movement, generating interactive graphs, and computing prediction accuracy.
- **`nn/nn_model_pipeline.py`** – Functions to train, test, and evaluate the neural network model for predicting player movements.
- **`utils.py`** – Utility functions for loading data, plotting graphs, and general helper methods.
- **`win_probability_model.py`** – Concept model used as an experiment and for progress report. Not used in final implementation.
