import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_metrics(bert_df, cnn_df):
    """
    Plots training and validation accuracy over epochs for BERT and CNN models.
    
    Args:
        bert_df (pd.DataFrame): DataFrame containing BERT training results.
        cnn_df (pd.DataFrame): DataFrame containing CNN training results.
    """
    # Filter out test rows and group by epoch to compute mean metrics
    bert_train = bert_df[bert_df['fold'] != 'test'].groupby('epoch')[['train_accuracy', 'validation_accuracy']].mean()
    cnn_train = cnn_df[cnn_df['fold'] != 'test'].groupby('epoch')[['train_accuracy', 'validation_accuracy']].mean()
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(bert_train.index, bert_train['train_accuracy'], label='BERT Train Accuracy', marker='o', color='#1f77b4')
    plt.plot(bert_train.index, bert_train['validation_accuracy'], label='BERT Validation Accuracy', marker='o', linestyle='--', color='#1f77b4')
    plt.plot(cnn_train.index, cnn_train['train_accuracy'], label='CNN Train Accuracy', marker='s', color='#ff7f0e')
    plt.plot(cnn_train.index, cnn_train['validation_accuracy'], label='CNN Validation Accuracy', marker='s', linestyle='--', color='#ff7f0e')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy: BERT vs CNN')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_metrics(bert_df, cnn_df):
    """
    Plots test metrics (accuracy, precision, recall, f1-score) for BERT and CNN models.
    
    Args:
        bert_df (pd.DataFrame): DataFrame containing BERT training results.
        cnn_df (pd.DataFrame): DataFrame containing CNN training results.
    """
    # Extract test metrics
    bert_test = bert_df[bert_df['fold'] == 'test'][['test_accuracy', 'precision', 'recall', 'f1_score']].iloc[0]
    cnn_test = cnn_df[cnn_df['fold'] == 'test'][['test_accuracy', 'precision', 'recall', 'f1_score']].iloc[0]
    
    # Create DataFrame for plotting
    test_metrics = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
        'Value': [
            bert_test['test_accuracy'], bert_test['precision'], bert_test['recall'], bert_test['f1_score'],
            cnn_test['test_accuracy'], cnn_test['precision'], cnn_test['recall'], cnn_test['f1_score']
        ],
        'Model': ['BERT'] * 4 + ['CNN'] * 4
    })
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot test metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=test_metrics, palette=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Test Metrics Comparison: BERT vs CNN')
    plt.legend(title='Model')
    plt.grid(True, axis='y')
    plt.savefig('test_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    bert_df = pd.read_csv('./bert_training_results.csv')
    cnn_df = pd.read_csv('./cnn_training_results.csv')
    
    # Plot metrics
    plot_training_metrics(bert_df, cnn_df)
    plot_test_metrics(bert_df, cnn_df)
    print("Visualizations saved as 'accuracy_comparison.png' and 'test_metrics_comparison.png'")

if __name__ == '__main__':
    main()