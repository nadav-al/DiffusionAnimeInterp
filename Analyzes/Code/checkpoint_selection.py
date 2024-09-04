import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict


def parse_test_folder_name(folder_name):
    parts = folder_name.split('_')
    rank = parts[2].replace('rk', '')
    cap = parts[-1] == "cap"
    return int(rank), cap


def load_csv_data(test_folder):
    csv_files = glob(os.path.join(test_folder, "*.csv"))
    data = []
    for file in csv_files:
        df = pd.read_csv(file)
        data.append(df)
    return pd.concat(data, axis=0)


def compute_mean_loss(data):
    return data.groupby('ckpt')['Frame2'].mean()


def get_similar_tests(all_tests):
    similar_tests = defaultdict(list)
    for test in all_tests:
        rank, cap = parse_test_folder_name(test)
        similar_tests[(rank, cap)].append(test)
    return similar_tests


def plot_mean_loss(test_folder, mean_losses, title):
    for test, mean_loss in mean_losses.items():
        plt.plot(mean_loss.index, mean_loss.values, label=test)
    plt.title(title)
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean Loss")
    plt.legend()
    plt.show()


def plot_all_scenes(scene_folder):
    all_tests = glob(os.path.join(scene_folder, "test*"))
    mean_losses = {}

    for test in all_tests:
        data = load_csv_data(test)
        mean_loss = compute_mean_loss(data)
        mean_losses[test] = mean_loss

    plot_mean_loss(scene_folder, mean_losses, f"Mean Loss of Each Checkpoint for Scene: {scene_folder}")


def plot_all_similar_tests(scene_folder):
    all_tests = glob(os.path.join(scene_folder, "test*"))
    similar_tests = get_similar_tests(all_tests)

    for (rank, cap), tests in similar_tests.items():
        mean_losses = {}
        for test in tests:
            data = load_csv_data(test)
            mean_loss = compute_mean_loss(data)
            mean_losses[test] = mean_loss
        plot_mean_loss(scene_folder, mean_losses,
                       f"Mean Loss for Similar Tests (rk{rank}_{cap}) across Scene: {scene_folder}")


def plot_all_tests_across_scenes(base_folder):
    all_tests = os.path.join(base_folder, "**/test*", recursive=True)
    mean_losses = {}

    for test in all_tests:
        data = load_csv_data(test)
        mean_loss = compute_mean_loss(data)
        mean_losses[test] = mean_loss

    plot_mean_loss(base_folder, mean_losses, "Mean Loss for All Tests Across All Scenes")


def plot_similar_tests_across_scenes(base_folder):
    all_tests = os.path.join(base_folder, "**/test*", recursive=True)
    similar_tests = get_similar_tests(all_tests)

    for (rank, cap), tests in similar_tests.items():
        mean_losses = {}
        for test in tests:
            data = load_csv_data(test)
            mean_loss = compute_mean_loss(data)
            mean_losses[test] = mean_loss
        plot_mean_loss(base_folder, mean_losses, f"Mean Loss for Similar Tests (rk{rank}_{cap}) Across All Scenes")


def suggest_top_checkpoints(base_folder, top_n=10):
    all_tests = os.path.join(base_folder, "**/test*", recursive=True)
    all_losses = []

    for test in all_tests:
        data = load_csv_data(test)
        mean_loss = compute_mean_loss(data)
        all_losses.append(mean_loss)

    combined_losses = pd.concat(all_losses, axis=1).mean(axis=1)
    top_checkpoints = combined_losses.nsmallest(top_n)
    print(f"Top {top_n} Checkpoints with Best Mean Loss:\n", top_checkpoints)
    return top_checkpoints


if __name__ == '__main__':
    # Example usage:
    base_folder = "/path/to/your/folder"
    plot_all_scenes(base_folder)
    plot_all_similar_tests(base_folder)
    plot_all_tests_across_scenes(base_folder)
    plot_similar_tests_across_scenes(base_folder)
    suggest_top_checkpoints(base_folder, top_n=10)






