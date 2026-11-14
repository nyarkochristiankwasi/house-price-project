from src.house_model import load_data, evaluate, save_model

df = load_data()
results = evaluate(df)
print("Test MAE:", results["test_mae"], "| R2:", results["test_r2"])
print("Cross-validated MAE:", results["cv_mae"])

save_model(df, results)
print("Saved app/house_price_model.joblib and app/model_info.json")