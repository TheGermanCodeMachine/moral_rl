import torch
from scipy.stats import pearsonr, spearmanr

def evaluate_mimic(model, test, labels):
    """Evaluate the model on the test set. Return the mean squared error."""
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        y_pred = model(test).squeeze()
        test_loss = loss(y_pred, labels).item()
        # Mean Squared Error
        print('test_loss', test_loss)
        # mean error
        print('test mean error', torch.mean(torch.abs(y_pred - labels)).item())
        # pearson correlation
        print('pearson correlation', pearsonr(y_pred, labels)[0])
        # spearman correlation
        print('spearman correlation', spearmanr(y_pred, labels)[0])
        
def print_weights(model):
    print('citizens_saved', model.weight[0][0].item())
    print('unsaved_citizens', model.weight[0][1].item())
    print('distance_to_citizen', model.weight[0][2].item())
    print('standing_on_extinguisher', model.weight[0][3].item())
    print('length', model.weight[0][4].item())
    print('bias', model.bias[0].item())