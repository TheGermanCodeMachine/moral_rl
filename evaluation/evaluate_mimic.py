import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from math import sqrt
import pickle

def print_inputs(inputs, features, model):
    weight = model.weight[0][0].item()
    input = inputs[0].item()
    print('citizens_saved', round(model.weight[0][0].item(), 2), ' * ', round(inputs[0].item(), 2), ' = ', round(model.weight[0][0].item() * inputs[0].item(), 2))
    print('unsaved_citizens', round(model.weight[0][1].item(), 2), ' * ', round(inputs[1].item(), 2), ' = ', round(model.weight[0][1].item() * inputs[1].item(), 2))
    print('distance_to_citizen', round(model.weight[0][2].item(), 2), ' * ', round(inputs[2].item(), 2), ' = ', round(model.weight[0][2].item() * inputs[2].item(), 2))
    print('standing_on_extinguisher', round(model.weight[0][3].item(), 2), ' * ', round(inputs[3].item(), 2), ' = ', round(model.weight[0][3].item() * inputs[3].item(), 2))
    print('length', round(model.weight[0][4].item(), 2), ' * ', round(inputs[4].item(), 2), ' = ', round(model.weight[0][4].item() * inputs[4].item(), 2))
    print('could_have_saved', round(model.weight[0][5].item(), 2), ' * ', round(inputs[5].item(), 2), ' = ', round(model.weight[0][5].item() * inputs[5].item(), 2))
    print('final_number_of_unsaved_citizens', round(model.weight[0][6].item(), 2), ' * ', round(inputs[6].item(), 2), ' = ', round(model.weight[0][6].item() * inputs[6].item(), 2))
    print('moved_towards_closest_citizen', round(model.weight[0][7].item(), 2), ' * ', round(inputs[7].item(), 2), ' = ', round(model.weight[0][7].item() * inputs[7].item(), 2))
    print('bias', round(model.bias[0].item(), 2))

# calculate loss metrics of the model on the test set
def evaluate_mimic(model, test, labels, print_it=False, worst=False, best=False, features=None):
    model.eval()
    """Evaluate the model on the test set. Return the mean squared error."""
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        y_pred = model(test).squeeze()
        # record multiple loss metrics
        test_loss = loss(y_pred, labels).item()
        pred_label_pairs = list(zip(y_pred, labels))
        test_mean_error = torch.mean(torch.abs(y_pred - labels)).item()
        rmse = sqrt(test_loss)
        r2 = r2_score(labels, y_pred)
        pearson_correlation = pearsonr(y_pred, labels)[0]
        spearman_correlation = spearmanr(y_pred, labels)[0]
        if print_it:
            print('test_loss', test_loss)
            print('test mean error', test_mean_error)
            print('pearson correlation', pearson_correlation)
            print('spearman correlation', spearman_correlation)

        if worst:
            # find the 3 predictions with the highest error
            errors = torch.abs(y_pred - labels)
            _, indices = torch.topk(errors, 1)
            print("worst predictions:", y_pred[indices[0]], "actual:", labels[indices[0]], "error:", errors[indices[0]])
            print_inputs(test[indices[0]], features, model)
        if best:
            # find the 3 predictions with the lowest error
            errors = torch.abs(y_pred - labels)
            _, indices = torch.topk(errors, 1, largest=False)
            print("best predictions:", y_pred[indices[0]], "actual:", labels[indices[0]], "error:", errors[indices[0]])
            print_inputs(test[indices[0]], features, model)
        
        return test_loss, test_mean_error, rmse, r2, pearson_correlation, spearman_correlation, pred_label_pairs
    
def evaluate_random(random_generator, test, labels):
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        y_pred = torch.tensor(random_generator.sample(test.shape[0]))
        # record multiple loss metrics
        test_loss = loss(y_pred, labels).item()
        pred_label_pairs = list(zip(y_pred, labels))
        test_mean_error = torch.mean(torch.abs(y_pred - labels)).item()
        rmse = sqrt(test_loss)
        r2 = r2_score(labels, y_pred)
        pearson_correlation = pearsonr(y_pred, labels)[0]
        spearman_correlation = spearmanr(y_pred, labels)[0]
        
        return test_loss, test_mean_error, rmse, r2, pearson_correlation, spearman_correlation, pred_label_pairs