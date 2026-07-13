from data_loader import load_data
from preprocess import split_and_scale
from train import find_best_k, train_model
from evaluate import evaluate_model

def main():
    df = load_data()
    class_names = sorted(df['target_name'].unique())
    print("Dataset loaded:", df.shape)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    best_k, error_rates = find_best_k(X_train, y_train)
    print(f"Best K found: {best_k}")

    model = train_model(X_train, y_train, n_neighbors=best_k)

    evaluate_model(model, X_test, y_test, class_names)

if __name__ == "__main__":
    main()