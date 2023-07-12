import torch
import matplotlib.pyplot as plt
import os

def show_loss_plot(train_losses, test_losses, show=True, lr=None, l2=None, base_path=None, epochs=None, save_path='', third_graph=None):

    # plot the training and test losses on a log scale
    #show labels
    plt.semilogy(test_losses, label='test' + 'lr={} l2={}'.format(lr, l2))
    plt.semilogy(train_losses, label='train' + 'lr={} l2={}'.format(lr, l2), linestyle='--')
    if isinstance(third_graph, list):
        plt.semilogy(third_graph, label='test_sin' + 'lr={} l2={}'.format(lr, l2), linestyle='--')
    # add a vertical line at epochs
    if epochs:
        plt.axvline(x=epochs, color='k', linestyle='--')
    if show:
        if l2:
            plt.title('lr', lr)
        plt.show()
    if save_path != '':
        #  if the results folder does not exist, create it
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path + 'loss_plot.png')    
        plt.close()

def print_example_predictions(model, test_set, test_labels):
    # print the first 5 predictions
    with torch.no_grad():
        y_pred_test = model(test_set).squeeze()
        for i in range(5):
            print('features', test_set[i], 'prediction', y_pred_test[i].item(), 'label', test_labels[i].item())

def get_weights(model, print_weights=False):
    weights = [model.weight[0][i].item() for i in range(len(model.weight[0]))]
    weights.append(model.bias[0].item())
    # weights = [model.weight[0][0].item(), model.weight[0][1].item(), model.weight[0][2].item(), model.weight[0][3].item(), model.weight[0][4].item(), model.weight[0][5].item(), model.weight[0][6].item(), model.weight[0][7].item(), model.bias[0].item()]
    if print_weights:
        print('citizens_saved', model.weight[0][0].item())
        print('unsaved_citizens', model.weight[0][1].item())
        print('distance_to_citizen', model.weight[0][2].item())
        print('standing_on_extinguisher', model.weight[0][3].item())
        print('length', model.weight[0][4].item())
        print('could_have_saved', model.weight[0][5].item())
        print('final_number_of_unsaved_citizens', model.weight[0][6].item())
        print('moved_towards_closest_citizen', model.weight[0][7].item())
        print('bias', model.bias[0].item())
    return weights