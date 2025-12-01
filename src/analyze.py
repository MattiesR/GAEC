import optuna
import optuna.visualization as viz
import argparse
import os

def analyze_study(storage_path: str, study_name: str):
    # Ensure the database exists
    if not os.path.exists(storage_path.replace("sqlite:///", "")):
        print(f"Database file does not exist: {storage_path}")
        return

    # Load the study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_path
    )

    print(f"Loaded study '{study_name}' with {len(study.trials)} trials")
    print(f"Best objective: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")

    # --- Visualizations ---
    print("Generating plots...")
    
    # 1. Optimization history
    fig_history = viz.plot_optimization_history(study)
    fig_history.show()

    # 2. Parallel coordinate plot
    fig_parallel = viz.plot_parallel_coordinate(study)
    fig_parallel.show()

    # 3. Slice plot
    fig_slice = viz.plot_slice(study)
    fig_slice.show()


    # 5. Parameter importance (optional)
    try:
        fig_importance = viz.plot_param_importances(study)
        fig_importance.show()
    except ImportError:
        print("scikit-learn not installed; skipping parameter importance plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna study from database")
    parser.add_argument("--db", type=str, default="sqlite:///gaec_study_tour50.db",
                        help="SQLite database path (sqlite:///file.db)")
    parser.add_argument("--study", type=str, default="gaec_study_tour50",
                        help="Study name to load")
    args = parser.parse_args()

    analyze_study(args.db, args.study)
