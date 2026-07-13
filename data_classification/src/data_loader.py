from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    """
    Loads the Iris dataset (150 samples, 3 classes, 4 features).
    Returns a pandas DataFrame with features + target column.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].apply(lambda i: iris.target_names[i])
    return df

if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print("\nShape:", data.shape)
    print("\nClasses:", data['target_name'].unique())