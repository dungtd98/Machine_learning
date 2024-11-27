import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load và chuẩn bị dữ liệu
def load_housing_data():
    # Load dữ liệu
    housing = pd.read_csv("housing.csv")
    return housing

def prepare_data(housing):
    # Xử lý missing values
    housing = housing.dropna()
    
    # Tách features và target
    X = housing.drop("median_house_value", axis=1)
    y = housing["median_house_value"]
    
    # Xử lý categorical variables
    X = pd.get_dummies(X, columns=["ocean_proximity"])
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 2. Tạo và huấn luyện mô hình
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 3. Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính các metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Housing Prices")
    plt.tight_layout()
    plt.show()
    
    return rmse, r2

# 4. Phân tích feature importance
def analyze_features(model, feature_names):
    # Get feature importance
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Sort by absolute value of coefficient
    coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(coefficients)), coefficients['Coefficient'])
    plt.xticks(range(len(coefficients)), coefficients['Feature'], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importance in Linear Regression Model')
    plt.tight_layout()
    plt.show()
    
    return coefficients

# 5. Main function
def main():
    # Load data
    housing = load_housing_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(housing)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # Analyze feature importance
    feature_names = housing.drop("median_house_value", axis=1).columns
    coefficients = analyze_features(model, feature_names)
    
    return model, scaler, coefficients

if __name__ == "__main__":
    model, scaler, coefficients = main()