import torch
import numpy as np
from torch import optim
from torch.nn import functional
from tqdm import tqdm
from testing.test_dataloaders import create_split_loaders
from testing.test_network import TestFeedforwardNet

# Database imports, TODO remove when possible

# extractor imports
from extraction.nn_data_extractor import NNDataExtractor


# Validation testing function
def test(model, computing_device, loader, criterion):
    total_val_loss = 0.0
    with torch.no_grad():
        for i, (val_images, val_labels) in enumerate(loader):
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
            #flatten images
            val_images = torch.reshape(val_images, (-1, 784))
            val_out = model(val_images)
            val_loss = criterion(val_out, val_labels)
            total_val_loss += float(val_loss)
    avg_val_loss = total_val_loss / i

    return total_val_loss, avg_val_loss


# Main training script
# Setup: initialize the hyper-parameters/variables
def main():

    extractor = NNDataExtractor()

    num_epochs = 50  # Number of full passes through the dataset
    early_stop_epochs = 5
    batch_size = 32  # Number of samples in each minibatch
    learning_rate = 0.001
    seed = np.random.seed(42)  # Seed the random number generator for reproducibility
    p_val = 0.1  # Percent of the overall dataset to reserve for validation
    p_test = 0.2  # Percent of the overall dataset to reserve for testing

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 3, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed,
                                                                 p_val=p_val, p_test=p_test,
                                                                 shuffle=True,
                                                                 extras=extras)

    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
    model = TestFeedforwardNet()
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    criterion = functional.cross_entropy

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Parameters: ", model.parameters())
    print("State dict: ", model.state_dict())

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    best_params = None

    # Save metadata about the training run.
    metadata = {"early_stop_epochs": early_stop_epochs, "seed": seed}

    extractor.extract_metadata(model_name="test_network", training_run_number=1, epochs=num_epochs,
                               batch_size=batch_size, cuda=use_cuda, model=model,
                               criterion=criterion, optimizer=optimizer, metadata=metadata)

    # extractor.extract_data(epoch=0, epoch_minibatch=0,
    #                        inputs=None, model_state=model.state_dict(),
    #                        outputs=None, targets=None)
    # was to record initial state but causes too many problems to be worth it. Also we gain very
    # little from recording it.
    # TODO remove or change. Maybe create empty Tensors for inputs and outputs?

    # Begin training procedure
    for epoch in range(num_epochs):

        N = 50
        N_minibatch_loss = 0.0
        current_best_val = 10000000.0
        increasing_epochs = 0

        # Get the next minibatch of images, labels for training
        torch.cuda.empty_cache()
        for minibatch_count, (images, labels) in tqdm(enumerate(train_loader, 0)):

            # Flatten images
            images = torch.reshape(images, (batch_size, -1))
            # print(images.shape)

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            soft_out = functional.softmax(outputs, dim=1)
            # print(soft_out.shape)

            extractor.extract_data(epoch=epoch+1, epoch_minibatch=minibatch_count+1,
                                   inputs=images, model_state=model.state_dict(),
                                   outputs=soft_out, targets=labels)
            print(minibatch_count)

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += float(loss)

            if minibatch_count % N == 0 and minibatch_count != 0:
                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                      (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

            # validate every 2 N minibatches.
            if minibatch_count % (2 * N) == 0 and minibatch_count != 0:

                # validation
                total_val_loss, avg_val_loss = test(model, computing_device, val_loader, criterion)
                if total_val_loss < current_best_val:
                    current_best_val = total_val_loss
                    best_params = model.state_dict()
                    increasing_epochs = 0
                else:
                    increasing_epochs += 1
                if increasing_epochs > early_stop_epochs:
                    break

                print(total_val_loss, avg_val_loss)

        print("Finished", epoch + 1, "epochs of training")

    print("Training complete after", epoch + 1, "epochs, with total loss: ", total_loss,
          " and average minibatch loss of: ", avg_minibatch_loss)

    if best_params is not None:
        model.load_state_dict(best_params)
    # test
    total_test_loss, avg_test_loss = test(model, computing_device, test_loader, criterion)
    print(total_test_loss, avg_test_loss)


if __name__ == '__main__':
    main()
