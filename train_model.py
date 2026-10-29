from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained model and scaler for the Streamlit app.
    joblib.dump(model, "iris_model.joblib")
    joblib.dump(scaler, "iris_scaler.joblib")
    print("\\nSaved iris_model.joblib and iris_scaler.joblib")

if __name__ == "__main__":
    main()
