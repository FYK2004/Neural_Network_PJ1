from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        self.training = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        grad_input = grad @ self.W.T
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Keep weight shape aligned with the starter comment.
        self.W = initialize_method(size=(1, out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))

        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}

        self.input = None
        self.input_padded = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    @staticmethod
    def _im2col(X, kernel_size, stride):
        """
        X: [N, C, H, W] (already padded if needed)
        return: X_col [N, out_h*out_w, C*k*k], out_h, out_w
        """
        N, C, H, W = X.shape
        k = kernel_size
        out_h = (H - k) // stride + 1
        out_w = (W - k) // stride + 1

        i0 = np.repeat(np.arange(k), k)
        i0 = np.tile(i0, C)
        j0 = np.tile(np.arange(k), k * C)
        i1 = stride * np.repeat(np.arange(out_h), out_w)
        j1 = stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        c = np.repeat(np.arange(C), k * k).reshape(-1, 1)
        cols = X[:, c, i, j]  # [N, C*k*k, out_h*out_w]
        cols = cols.transpose(0, 2, 1)  # [N, out_h*out_w, C*k*k]
        return cols, out_h, out_w

    @staticmethod
    def _col2im(cols, X_shape, kernel_size, stride):
        """
        cols: [N, out_h*out_w, C*k*k]
        X_shape: (N, C, H, W) of padded input
        return: dX [N, C, H, W]
        """
        N, C, H, W = X_shape
        k = kernel_size
        out_h = (H - k) // stride + 1
        out_w = (W - k) // stride + 1

        cols = cols.transpose(0, 2, 1)  # [N, C*k*k, out_h*out_w]

        i0 = np.repeat(np.arange(k), k)
        i0 = np.tile(i0, C)
        j0 = np.tile(np.arange(k), k * C)
        i1 = stride * np.repeat(np.arange(out_h), out_w)
        j1 = stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        c = np.repeat(np.arange(C), k * k).reshape(-1, 1)

        X = np.zeros((N, C, H, W), dtype=cols.dtype)
        np.add.at(X, (slice(None), c, i, j), cols)
        return X

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        if self.padding > 0:
            self.input_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            self.input_padded = X

        X_col, out_h, out_w = self._im2col(self.input_padded, self.kernel_size, self.stride)
        # cache for backward
        self._X_col = X_col
        self._out_h = out_h
        self._out_w = out_w

        W_col = self.W.reshape(self.out_channels, -1)  # [out, C*k*k]
        out = X_col @ W_col.T  # [N, out_h*out_w, out]
        out = out + self.b.reshape(1, 1, self.out_channels)
        out = out.transpose(0, 2, 1).reshape(X.shape[0], self.out_channels, out_h, out_w)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        N, _, out_h, out_w = grads.shape
        grads_2d = grads.transpose(0, 2, 3, 1).reshape(N, out_h * out_w, self.out_channels)  # [N, P, out]

        X_col = self._X_col  # [N, P, C*k*k]
        W_col = self.W.reshape(self.out_channels, -1)  # [out, C*k*k]

        # dB
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1)

        # dW: sum over batch and spatial positions
        dW_col = np.tensordot(grads_2d, X_col, axes=([0, 1], [0, 1]))  # [out, C*k*k]
        dW = dW_col.reshape(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # dX
        dX_col = grads_2d @ W_col  # [N, P, C*k*k]
        dX_padded = self._col2im(dX_col, self.input_padded.shape, self.kernel_size, self.stride)

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class Dropout(Layer):
    """
    Dropout for fully-connected activations: input/output shapes are identical.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = float(p)
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training or self.p <= 0:
            return X

        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*X.shape) < keep_prob).astype(X.dtype)
        return (X * self.mask) / keep_prob

    def backward(self, grads):
        if not self.training or self.p <= 0:
            return grads

        keep_prob = 1.0 - self.p
        return (grads * self.mask) / keep_prob


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True

        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels

        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        self.probs = probs

        batch_size = predicts.shape[0]
        one_hot = np.zeros((batch_size, self.max_classes))
        one_hot[np.arange(batch_size), labels] = 1

        eps = 1e-12
        loss = -np.sum(one_hot * np.log(probs + eps)) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        one_hot = np.zeros((batch_size, self.max_classes))
        one_hot[np.arange(batch_size), self.labels] = 1

        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:
            eps = 1e-12
            self.grads = -(one_hot / (self.probs + eps)) / batch_size

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition