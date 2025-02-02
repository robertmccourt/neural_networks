import numpy as np

from activations import ReLU, BatchNormalize
from CNN import Conv3d, MaxPool2d
from FNN.fnn import FNN
from FNN.layer import Layer as FFLayer

class CNN:
    """
    Convolutional Neural Network.
    """

    def __init__(self, input_shape, num_classes):
        """
        Initialize the CNN.
        """
        channels, height, width = input_shape

        # Use dynamic in_channels for CIFAR (3 channels) or MNIST (1 channel)
        self.c1 = Conv3d(in_channels=channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.r1 = ReLU()
        self.b1 = BatchNormalize()
        self.s2 = MaxPool2d(kernel_size=2, stride=2)
        
        
        self.c3 = Conv3d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.r3 = ReLU()
        self.b3 = BatchNormalize()
        self.s4 = MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = FFLayer(n_input=400, n_output=120, activation='relu')
        self.b4 = BatchNormalize()
        self.fc2 = FFLayer(n_input=120, n_output=84, activation='relu')
        self.b5 = BatchNormalize()
        self.fc3 = FFLayer(n_input=84, n_output=num_classes, activation='logsoftmax')  

        self.layers = [self.c1, self.r1, self.s2, self.c3, self.r3, self.s4, self.fc1, self.fc2, self.fc3]        
        self.output = None

    def forward(self, x):
        """
        Forward pass for LeNet.
        """
        # Transpose input to NCHW format (batch_size, channels, height, width)
        #x = np.transpose(x, (0, 3, 1, 2))

        # Layer 1: Convolution -> ReLU -> Max Pooling
        x_conv1 = self.c1.forward(x)  # Convolution
        x = self.r1.forward(x_conv1)  # ReLU activation
        x = self.b1.forward(x)
        x = self.s2.forward(x)        # Max Pooling

        # Save input for backpropagation (only input to the convolution is needed)
        #self.c1.input = x_conv1

        # Layer 2: Convolution -> ReLU -> Max Pooling
        x_conv2 = self.c3.forward(x)  # Convolution
        x = self.r3.forward(x_conv2)  # ReLU activation
        x = self.b3.forward(x)
        x = self.s4.forward(x)        # Max Pooling

        # Save input for backpropagation
        # self.c3.input = x_conv2

        # Flatten for Fully Connected Layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Fully Connected Layers
        x = self.fc1.forward(x)
        x = self.b4.forward(x)
        x = self.fc2.forward(x)
        x = self.b5.forward(x)
        x = self.fc3.forward(x)


        return x



    def backward(self, y, y_pred, learning_rate, loss_func ="nll"):
        """
        Perform backpropagation through all layers of the CNN
        params:
            dL_dout: Gradient of the loss with respect to the output of the CNN (shape: batch_size x num_classes).
            learning_rate: Learning rate.
        """
        if loss_func == 'mse':
            #Regular
            dL_dout = 2 * (y_pred - y) / y.shape[0]
        elif loss_func == 'nll':
            p = np.exp(y_pred)
            dL_dout = p - y

        # Feed Forward Layers
        grad_W, dL_dout = self.fc3.backward(dL_dout)
        print("Mean abs grad FC3 weights:", np.mean(np.abs(grad_W)))
        grad_W = np.clip(grad_W, -5, 5)
        self.fc3.weights -= learning_rate * grad_W

        grad_W, dL_dout = self.fc2.backward(dL_dout)
        print("Mean abs grad FC2 weights:", np.mean(np.abs(grad_W)))
        grad_W = np.clip(grad_W, -5, 5)
        self.fc2.weights -= learning_rate * grad_W

        grad_W, dL_dout = self.fc1.backward(dL_dout)
        print("Mean abs grad FC1 weights:", np.mean(np.abs(grad_W)))
        grad_W = np.clip(grad_W, -5, 5)
        self.fc1.weights -= learning_rate * grad_W

        # Begin CNN layers

        # reshape for max pool?
        batch_size, original_channels, height, width = self.s4.output.shape
        dL_dout = dL_dout.reshape(batch_size, original_channels, height, width)
        dL_dout = self.s4.backward(dL_dout)

        # conv
        dL_dout, grad_filters, grad_biases = self.c3.backward(dL_dout)
        grad_filters = np.clip(grad_filters, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.c3.filters -= learning_rate * grad_filters
        self.c3.biases -= learning_rate * grad_biases 
        print(f"shape of grad filters and biases: {grad_filters.shape}, {grad_biases.shape}")

        # maxpool
        dL_dout = self.s2.backward(dL_dout)

        # conv
        dL_dout, grad_filters, grad_biases = self.c1.backward(dL_dout)
        grad_filters = np.clip(grad_filters, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.c1.filters -= learning_rate * grad_filters
        self.c1.biases -= learning_rate * grad_biases 


    def backward_adam(self, y, y_pred, t, learning_rate, rho=0.999, rho_f=0.9, epsilon=1e-8, loss_func ="nll"):
        """
        Perform backpropagation through all layers of the CNN
        params:
            dL_dout: Gradient of the loss with respect to the output of the CNN (shape: batch_size x num_classes).
            learning_rate: Learning rate.
        """
        
        alpha_t = learning_rate * ((np.sqrt(1 - (rho ** t))) / (1 - (rho_f ** t) + 1e-8))
        print(f"alpha t {alpha_t}")

        if loss_func == 'mse':
            #Regular
            dL_dout = 2 * (y_pred - y) / y.shape[0]
        elif loss_func == 'nll':
            p = np.exp(y_pred)
            dL_dout = p - y
        elif loss_func == "cross_entropy":
            dL_dout = -np.sum(y * y_pred ) / y.shape[0]
        # print(dL_dout)

        # Feed Forward Layers
        grad_W, dL_dout = self.fc3.backward(dL_dout)
        grad_W = np.clip(grad_W, -5, 5)
        self.fc3.update_A(grad_W, rho)
        self.fc3.update_F(grad_W, rho_f)
        adaptive_step = self.getAdaptiveStep(self.fc3.A, 
                                             self.fc3.F, 
                                             alpha_t, rho, 
                                             rho_f,
                                             t, epsilon)
        self.fc3.weights -= adaptive_step

        grad_W, dL_dout = self.fc2.backward(dL_dout)
        grad_W = np.clip(grad_W, -5, 5)
        self.fc2.update_A(grad_W, rho)
        self.fc2.update_F(grad_W, rho_f)
        adaptive_step = self.getAdaptiveStep(self.fc2.A, 
                                             self.fc2.F, 
                                             alpha_t, rho, 
                                             rho_f,
                                             t, epsilon)
        self.fc2.weights -= adaptive_step

        grad_W, dL_dout = self.fc1.backward(dL_dout)
        grad_W = np.clip(grad_W, -5, 5)
        self.fc1.update_A(grad_W, rho)
        self.fc1.update_F(grad_W, rho_f)
        adaptive_step = self.getAdaptiveStep(self.fc1.A, 
                                             self.fc1.F, 
                                             alpha_t, rho, 
                                             rho_f,
                                             t, epsilon)
        self.fc1.weights -= adaptive_step

        # Begin CNN layers

        # reshape for max pool?
        batch_size, original_channels, height, width = self.s4.output.shape
        dL_dout = dL_dout.reshape(batch_size, original_channels, height, width)
        dL_dout = self.s4.backward(dL_dout)

        # conv
        dL_dout, grad_filters, grad_biases = self.c3.backward(dL_dout)
        grad_filters = np.clip(grad_filters, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.c3.update_A(grad_filters, rho)
        self.c3.update_F(grad_filters, rho_f)
        adaptive_step = self.getAdaptiveStep(self.c3.A, 
                                             self.c3.F, 
                                             alpha_t, rho, 
                                             rho_f,
                                             t, epsilon)
        self.c3.filters -= adaptive_step
        self.c3.biases -= learning_rate * grad_biases 
        print(f"shape of grad filters and biases: {grad_filters.shape}, {grad_biases.shape}")

        # maxpool
        dL_dout = self.s2.backward(dL_dout)

        # conv
        dL_dout, grad_filters, grad_biases = self.c1.backward(dL_dout)
        grad_filters = np.clip(grad_filters, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.c1.update_A(grad_filters, rho)
        self.c1.update_F(grad_filters, rho_f)
        adaptive_step = self.getAdaptiveStep(self.c1.A, 
                                             self.c1.F, 
                                             alpha_t, rho, 
                                             rho_f,
                                             t, epsilon)
        self.c1.filters -= adaptive_step
        self.c1.biases -= learning_rate * grad_biases


    def train(self, input, labels, epochs: int, learning_rate: int = 0.01, optimizer = "gd"):
        input = (input - 0.5) / 0.5
        training_accuracy = []
        for epoch in range(epochs):
            loss_sum = 0  # accumulated loss within epoch
            correct = 0   # correctly classified samples
            total = 0     # total samples

            # Forward pass
            out = self.forward(input)

            # Convert raw class indices to one-hot encoding
            if labels.ndim == 1:  # If labels are raw class indices
                num_classes = out.shape[1]
                labels = np.eye(num_classes)[labels]  # Convert to one-hot

            # Backward pass
            if optimizer == "gd":
                self.backward(labels, out, learning_rate, loss_func="nll")
            elif optimizer == "adam":
                self.backward_adam(labels, out, epoch+1, learning_rate, loss_func="nll")
            else:
                raise ValueError(f"{optimizer} not an available optimizer")


            # Calculate predictions and accuracy
            predictions = np.argmax(out, axis=1)
            true_labels = np.argmax(labels, axis=1)
            correct += np.sum(predictions == true_labels)
            total += labels.shape[0]
            accuracy = (correct / total) * 100

            loss = -np.mean(np.sum(labels * out, axis=1))

            # Print accuracy after each epoch
            print(f"Epoch {epoch + 1}/{epochs}- Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
            training_accuracy.append(accuracy)

            if accuracy > 95:
                print("Reached high enough accuracy")
                break
        
        return training_accuracy


    def getAccuracy(self, test_loader):
        correct = 0
        total = 0

        for idx, test_batch in enumerate(test_loader):
            test_input, test_labels = test_batch
            test_input = test_input.numpy()
            test_labels = test_labels.numpy()

            # Normalize input
            test_input = (test_input - 0.5) / 0.5  # Match normalization from training

            # Forward pass
            out = self.forward(test_input)

            # Predictions
            predicted_classes = np.argmax(out, axis=1)  # Predicted class indices
            true_classes = test_labels  # Directly use 1D array of true class indices

            # Accuracy calculation
            correct += np.sum(predicted_classes == true_classes)
            total += test_labels.shape[0]

            # Break early for debugging
            if idx > 5:  # Limit to 5 batches
                break

        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def clip_gradient(self, gradient, threshold=1.0):
        norm = np.linalg.norm(gradient)
        if norm > threshold:
            scaling_factor = np.clip(threshold / norm, a_min=0, a_max=1.0)

            # Scale the gradient using the factor
            gradient = gradient * scaling_factor

        # print(gradient) 
        return gradient


    def getAdaptiveStep(self, A, F, alpha_t, rho, rho_f, t, epsilon):
            A_hat = A * (1 / ((1 - (rho ** t)) + 1e-8))
            F_hat = F * (1 / ((1 - (rho_f ** t)) + 1e-8))
            adaptive_step = alpha_t * F_hat / (np.sqrt(A_hat) + epsilon)
            return adaptive_step 