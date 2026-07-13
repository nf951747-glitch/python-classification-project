from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale(df, test_size=0.2, random_state=42):
    """
    Splits data into train/test sets and applies StandardScaler
    (mean=0, variance=1) so no single feature dominates due to scale.
    """
    X = df.drop(columns=['target', 'target_name'])
    y = df['target']

    # Shuffle happens by default in train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, don't fit again

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler