import torch
from scipy.stats import pearsonr, spearmanr

def evaluate_mimic(model, test, labels, print=False):
    """Evaluate the model on the test set. Return the mean squared error."""
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        y_pred = model(test).squeeze()
        test_loss = loss(y_pred, labels).item()
        # Mean Squared Error
        # mean error
        test_mean_error = torch.mean(torch.abs(y_pred - labels)).item()
        # pearson correlation
        pearson_correlation = pearsonr(y_pred, labels)[0]
        # spearman correlation
        spearman_correlation = spearmanr(y_pred, labels)[0]
        if print:
            print('test_loss', test_loss)
            print('test mean error', test_mean_error)
            print('pearson correlation', pearson_correlation)
            print('spearman correlation', spearman_correlation)
        return test_loss, test_mean_error, pearson_correlation, spearman_correlation
    