import numpy as np
from FNN.layer import Layer

class Conv3d:
    """
    3D Convolutional Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)

        # for adam optimization
        self.A = np.zeros_like(self.filters)
        self.F = np.zeros_like(self.filters)
        
        #print(f"Initialized filters: {self.filters.shape}, biases: {self.biases.shape}")


    def pad_matrix(self, input, pad_size):
        return np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')


    def forward(self, input):
        """
        params:
            input: 4D input array of shape (batch_size, in_channels, height, width).
        return:
            conv3d output
        """
        self.input = input
        print("from forward, input shape: ", input.shape)
        return self.conv3d(input, self.filters, self.biases, self.stride, self.padding)

    def conv2d(self, input_channel, kernel, bias, stride):
        """
        params:
            input_channel: one of the 2D input channels (height x width).
            kernel: the associated 2D kernel (height x width).
            bias: bias for the kernel
            stride: stride of convolution.

        return:
            2D output matrix.
        """
        
        input_height, input_width = input_channel.shape
        kernel_height, kernel_width = kernel.shape

        # output dimensions
        out_height = (input_height - kernel_height) // stride + 1
        out_width = (input_width - kernel_width) // stride + 1


        output = np.zeros((out_height, out_width))

        # 2D convolution
        for i in range(out_height):
            for j in range(out_width):
                region = input_channel[
                         i * stride:i * stride + kernel_height,
                         j * stride:j * stride + kernel_width
                         ]

                region = np.clip(region, -1e3, 1e3)
                kernel = np.clip(kernel, -1e3, 1e3)

            # Perform convolution
                try:
                    output[i][j] = np.sum(region * kernel) + bias
                except RuntimeWarning as e:
                    print(f"Overflow in convolution at position ({i}, {j}):", e)
                    print("Region:", region)
                    print("Kernel:", kernel)
                    raise e
        
        # Debug: Print output of the convolution
        #print(f"Convolution output shape: {output.shape}")
        
        return output
    
    def conv3d(self, input, filters, biases, stride=1, padding=0):
        """
        params:
            input: 4D input array of shape (batch_size x input_channels x height x width).
            filters: filters (num_filters x input_channels x kernel_height x kernel_width).
            biases: bias for each filter
            stride: stride of convolution.
            padding: padding on height and width.

        return:
            4D output matrix (batch_size x num_filters x out_height x out_width).
        """
        
        # print(f"conv3d - input shape: {input.shape}")
        """
        if padding > 0:
            self.input = self.pad_matrix(input, padding)
        else:
            self.input = input
            """

        batch_size, input_channels, input_height, input_width = input.shape
        num_filters, filter_channels, kernel_height, kernel_width = filters.shape
        
        # print the input shape
        # print(f"Input shape: {input.shape}")
        # print("batch size: ", batch_size)
        # print("input height: ", input_height)
        # print("input channels: ", input_channels)
        # print("input width: ", input_width)
        
        # print(f"Filters shape: {filters.shape}")
        # print("num filters: ", num_filters)
        # print("filter channels: ", filter_channels)
        # print("kernel height: ", kernel_height)
        # print("kernel width: ", kernel_width)

        # output dimensions
        out_height = (input_height - kernel_height + 2 * padding) // stride + 1
        out_width = (input_width - kernel_width + 2 * padding) // stride + 1
        
        # Debug: Print calculated output dimensions
        # print(f"Calculated output dimensions: out_height={out_height}, out_width={out_width}")

        # output initialization
        output = np.zeros((batch_size, num_filters, out_height, out_width))

        # 3D convolution
        for batch in range(batch_size):
            for filter in range(num_filters):
                global_output = np.zeros((out_height, out_width))
                for channel in range(input_channels):
                    # Sum all channels
                    global_output += self.conv2d(
                        input[batch, channel], filters[filter, channel], biases[filter], stride
                    )
                output[batch, filter] = global_output
                
        return output
    

    def backward(self, dL_dout):
        """
        params:
            dL_dout: derivative of loss w.r.t output of previous layer
            
        return:
        """
        batch_size, dL_dout_channels, dL_dout_height, dL_dout_width = dL_dout.shape 
        num_filters, in_channels, kernel_height, kernel_width = self.filters.shape
        dL_db = np.zeros(num_filters)

        # compute dL \ dF
        dL_df = np.zeros_like(self.filters)
        for batch in range(batch_size):
            for filter in range(num_filters):
                for channel in range(in_channels):
                    dL_df[filter, channel, :,:] = self.conv2d(self.input[batch, channel,:,:],
                                                    dL_dout[batch, filter,:,:],
                                                    0, # bias is 0 because we don't need it here
                                                    stride=self.stride)
                
        # computer dL \ dx (this is the part that will be backpropogated)
        dL_dx = np.zeros_like(self.input)
        print("input shape ", self.input.shape)
        for batch in range(batch_size):
            for filter in range(num_filters):
                for channel in range(in_channels):
                    # rotate 90 degrees twice
                    rotated_filter = np.rot90(self.filters[filter,channel,:,:], 
                                            k=2,)
                    
                    # Add padding to dL_dout to match input size (padding = kernel size - 1)
                    # Slides say "Notice that you need to extend the matrix."
                    # print(self.filters[filter,channel,:,:])
                    pad_size = (self.filters.shape[2] - 1) #// 2
                    # print(pad_size)
                    padded_dL_dout = self.pad_matrix(dL_dout, pad_size)
                    # print(padded_dL_dout.shape)
                    dL_dx[batch,channel,:,:] = self.conv2d(padded_dL_dout[batch, filter,:,:],
                                                    rotated_filter,
                                                    0, # bias is zero because we handle is separately
                                                    stride=self.stride)
                    
        # compute bias gradient
        for filter in range(num_filters):
            dL_db[filter] = np.sum(dL_dout[:, filter, :, :]) 


        return dL_dx, dL_df, dL_db
    
    def update_A(self, grad_filter, rho=0.999):
        grad_filter = np.clip(grad_filter, -3, 3)
        num_filters, in_channels, kernel_height, kernel_width = grad_filter.shape
        for filter in range(num_filters):
            for channel in range(in_channels):
                self.A[filter, channel, :, :]  = (rho * self.A[filter, channel, :, :] 
                                                  + (1 - rho) * (grad_filter[filter, channel, :, :]**2)) 

            
    def update_F(self, grad_filter, rho_f=0.9):
        grad_filter = np.clip(grad_filter, -3, 3)
        num_filters, in_channels, kernel_height, kernel_width = grad_filter.shape
        for filter in range(num_filters):
            for channel in range(in_channels):
                self.F[filter, channel, :, :]  = (rho_f * self.F[filter, channel, :, :] 
                                                  + (1 - rho_f) * (grad_filter[filter, channel, :, :]))
                



class MaxPool2d:
    """
    2D Max Pooling Layer.
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.max_indices = None  # To store indices of max values during forward pass

    def forward(self, input):
        """
        Perform max pooling and store argmax indices.
        """
        self.input = input
        batch_size, channels, H_in, W_in = input.shape

        # Calculate the output dimensions
        H_out = ((H_in - self.kernel_size) // self.stride) + 1
        W_out = ((W_in - self.kernel_size) // self.stride) + 1

        # Initialize output and indices
        output = np.zeros((batch_size, channels, H_out, W_out))
        self.max_indices = np.zeros((batch_size, channels, H_out, W_out, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        # Find max and store it
                        max_idx_flat = np.argmax(window)
                        max_idx = np.unravel_index(max_idx_flat, (self.kernel_size, self.kernel_size))
                        
                        output[b, c, i, j] = window[max_idx]
                        # Store the exact indices in the original input
                        self.max_indices[b, c, i, j, 0] = h_start + max_idx[0]
                        self.max_indices[b, c, i, j, 1] = w_start + max_idx[1]

        self.output = output
        return output
    
    def backward(self, dL_dout):
        """
        Backpropagation using stored max indices.
        """
        batch_size, channels, H_in, W_in = self.input.shape
        _, _, H_out, W_out = dL_dout.shape

        # Initialize gradient w.r.t input
        dL_dinput = np.zeros_like(self.input)

        # Directly use the stored max indices
        for b in range(batch_size):
            for c in range(channels):
                for i in range(H_out):
                    for j in range(W_out):
                        max_i, max_j = self.max_indices[b, c, i, j]
                        dL_dinput[b, c, max_i, max_j] += dL_dout[b, c, i, j]

        return dL_dinput
