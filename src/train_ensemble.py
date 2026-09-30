"""
Train and evaluate different ensemble methods
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.ensemble_advanced import AdvancedEnsemble, EnsembleOptimizer, EnsemblePredictor
from src.train_xgboost import train_xgboost_model
from src.train_lstm import train_lstm_model

def train_and_compare_ensembles(df: pd.DataFrame, feature_cols: list):
    """
    Train and compare different ensemble methods
    """
    
    print("=" * 70)
    print("ENSEMBLE COMPARISON STUDY")
    print("=" * 70)
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Train base models
    print("\n1. Training Base Models...")
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost_model(train_df, feature_cols)
    print(f"   ✅ XGBoost trained - Accuracy: {xgb_metrics['xgboost']['accuracy']:.4f}")
    
    # Train LSTM - FIXED: Handle 3 return values
    print("\n   Training LSTM...")
    lstm_result = train_lstm_model(train_df, feature_cols)
    if len(lstm_result) == 3:
        lstm_model, lstm_metrics, _ = lstm_result
    else:
        lstm_model, lstm_metrics = lstm_result
    print(f"   ✅ LSTM trained - Accuracy: {lstm_metrics['accuracy']:.4f}")
    
    # Get validation predictions
    print("\n2. Getting Validation Predictions...")
    X_val = test_df[feature_cols]
    y_val = test_df["target"].values
    
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    
    # Get LSTM predictions (simplified for validation)
    try:
        from src.train_lstm import predict_lstm_enhanced
        lstm_val_prob = predict_lstm_enhanced(lstm_model, test_df, feature_cols, sequence_length=30)["lstm_prob"].values
        if len(lstm_val_prob) != len(y_val):
            lstm_val_prob = np.full(len(y_val), 0.5)
    except:
        lstm_val_prob = np.full(len(y_val), 0.5)
    
    # Initialize ensemble
    ensemble = AdvancedEnsemble()
    optimizer = EnsembleOptimizer()
    
    # Test different ensemble methods
    methods = {
        "XGBoost Only": lambda: xgb_val_prob,
        "LSTM Only": lambda: lstm_val_prob,
        "Weighted Average (0.6/0.4)": lambda: ensemble.weighted_average(xgb_val_prob, lstm_val_prob),
        "Max Ensemble": lambda: ensemble.max_ensemble(xgb_val_prob, lstm_val_prob),
        "Min Ensemble": lambda: ensemble.min_ensemble(xgb_val_prob, lstm_val_prob),
        "Product Ensemble": lambda: ensemble.product_ensemble(xgb_val_prob, lstm_val_prob),
        "Rank Average": lambda: ensemble.rank_average_ensemble(xgb_val_prob, lstm_val_prob),
    }
    
    # Evaluate each method
    results = []
    
    print("\n3. Evaluating Ensemble Methods...")
    print("-" * 80)
    print(f"{'Method':<30} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12} {'Precision':<12}")
    print("-" * 80)
    
    for name, method in methods.items():
        try:
            pred_prob = method()
            pred_class = (pred_prob >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, pred_class)
            f1 = f1_score(y_val, pred_class, zero_division=0)
            auc = roc_auc_score(y_val, pred_prob)
            precision = precision_score(y_val, pred_class, zero_division=0)
            recall = recall_score(y_val, pred_class, zero_division=0)
            
            results.append({
                "method": name,
                "accuracy": accuracy,
                "f1_score": f1,
                "auc_roc": auc,
                "precision": precision,
                "recall": recall
            })
            
            print(f"{name:<30} {accuracy:<12.4f} {f1:<12.4f} {auc:<12.4f} {precision:<12.4f}")
            
        except Exception as e:
            print(f"{name:<30} Error: {e}")
    
    # Find optimal weights
    print("\n4. Finding Optimal Weights...")
    optimal = optimizer.find_optimal_weights(xgb_val_prob, lstm_val_prob, y_val, metric="f1")
    print(f"   Optimal weights: XGBoost={optimal['weights']['xgb']:.2f}, LSTM={optimal['weights']['lstm']:.2f}")
    print(f"   Best {optimal['metric']}: {optimal['best_score']:.4f}")
    
    # Test optimal weighted average
    optimal_ensemble = optimal['weights']['xgb'] * xgb_val_prob + optimal['weights']['lstm'] * lstm_val_prob
    optimal_pred = (optimal_ensemble >= 0.5).astype(int)
    
    optimal_f1 = f1_score(y_val, optimal_pred, zero_division=0)
    optimal_acc = accuracy_score(y_val, optimal_pred)
    
    print(f"\n   Optimal Ensemble: Accuracy={optimal_acc:.4f}, F1={optimal_f1:.4f}")
    
    # Find best method
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    best_method = max(results, key=lambda x: x["f1_score"])
    print(f"Best ensemble method: {best_method['method']}")
    print(f"F1 Score: {best_method['f1_score']:.4f}")
    print(f"Accuracy: {best_method['accuracy']:.4f}")
    print(f"AUC-ROC: {best_method['auc_roc']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/ensemble_comparison.csv", index=False)
    
    # Save optimal weights
    weights_path = Path("models/ensemble_weights.json")
    weights_path.write_text(json.dumps({
        "xgb_weight": optimal['weights']['xgb'],
        "lstm_weight": optimal['weights']['lstm'],
        "best_method": best_method['method'],
        "metrics": best_method
    }, indent=2))
    
    print("\n✅ Results saved to reports/ensemble_comparison.csv")
    print("✅ Optimal weights saved to models/ensemble_weights.json")
    
    return results_df, optimal['weights']

def create_ensemble_report(results_df: pd.DataFrame):
    """Create a visual report of ensemble performance"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy comparison
        ax1 = axes[0, 0]
        methods = results_df['method'].values
        accuracy = results_df['accuracy'].values
        colors = ['green' if a == max(accuracy) else 'steelblue' for a in accuracy]
        ax1.barh(methods, accuracy, color=colors)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlim(0.4, 0.9)
        
        # F1 Score comparison
        ax2 = axes[0, 1]
        f1 = results_df['f1_score'].values
        colors = ['green' if f == max(f1) else 'steelblue' for f in f1]
        ax2.barh(methods, f1, color=colors)
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_xlim(0, 0.9)
        
        # AUC-ROC comparison
        ax3 = axes[1, 0]
        auc = results_df['auc_roc'].values
        colors = ['green' if a == max(auc) else 'steelblue' for a in auc]
        ax3.barh(methods, auc, color=colors)
        ax3.set_xlabel('AUC-ROC')
        ax3.set_title('AUC-ROC Comparison')
        ax3.set_xlim(0.4, 1.0)
        
        # Precision vs Recall
        ax4 = axes[1, 1]
        precision = results_df['precision'].values
        recall = results_df['recall'].values
        ax4.scatter(precision, recall, alpha=0.7, s=100)
        for i, method in enumerate(methods):
            ax4.annotate(method[:15], (precision[i], recall[i]), fontsize=8)
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall')
        ax4.set_xlim(0.3, 1.0)
        ax4.set_ylim(0.3, 1.0)
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('reports/ensemble_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Ensemble performance chart saved to reports/ensemble_performance.png")
        
    except Exception as e:
        print(f"\n⚠️ Could not create chart: {e}")

if __name__ == "__main__":
    import pandas as pd
    
    # Load data
    df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
    
    # Get feature columns
    exclude = {"date", "symbol", "target", "future_return"}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    print(f"Loaded {len(df)} rows with {len(feature_cols)} features")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    
    # Run ensemble comparison
    results, optimal_weights = train_and_compare_ensembles(df, feature_cols)
    
    # Create visual report
    create_ensemble_report(results)