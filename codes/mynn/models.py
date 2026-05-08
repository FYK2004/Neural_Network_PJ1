from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=0.0):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_p = float(dropout_p)

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout_p > 0:
                        self.layers.append(Dropout(self.dropout_p))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        offset = 2
        self.dropout_p = 0.0
        if len(param_list) > offset and isinstance(param_list[offset], (int, float)):
            self.dropout_p = float(param_list[offset])
            offset += 1

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + offset]['W']
                layer.b = param_list[i + offset]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + offset]['weight_decay']
                layer.weight_decay_lambda = param_list[i + offset]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout_p > 0:
                        self.layers.append(Dropout(self.dropout_p))
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.dropout_p]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, dropout_p=0.0):
        # CNN for MNIST (two conv layers).
        self.conv_out_channels = 8
        self.conv_kernel_size = 3
        self.conv_stride = 2
        self.conv_padding = 1
        self.conv2_out_channels = 8
        self.conv2_kernel_size = 3
        self.conv2_stride = 1
        self.conv2_padding = 1
        self.hidden_dim = 128
        self.num_classes = 10
        self.dropout_p = float(dropout_p)

        # Input size is fixed to MNIST 28x28 in this project.
        out_h = (28 + 2 * self.conv_padding - self.conv_kernel_size) // self.conv_stride + 1
        out_w = out_h
        out_h2 = (out_h + 2 * self.conv2_padding - self.conv2_kernel_size) // self.conv2_stride + 1
        out_w2 = (out_w + 2 * self.conv2_padding - self.conv2_kernel_size) // self.conv2_stride + 1
        conv_feat_dim = self.conv2_out_channels * out_h2 * out_w2

        self.layers = [
            conv2D(
                in_channels=1,
                out_channels=self.conv_out_channels,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
            ),
            ReLU(),
            conv2D(
                in_channels=self.conv_out_channels,
                out_channels=self.conv2_out_channels,
                kernel_size=self.conv2_kernel_size,
                stride=self.conv2_stride,
                padding=self.conv2_padding,
            ),
            ReLU(),
            Linear(in_dim=conv_feat_dim, out_dim=self.hidden_dim),
            ReLU(),
        ]
        if self.dropout_p > 0:
            self.layers.append(Dropout(self.dropout_p))
        self.layers.append(Linear(in_dim=self.hidden_dim, out_dim=self.num_classes))
        self._conv_feature_shape = None

        # He initialization + zero bias for stable training.
        conv1 = self.layers[0]
        fan_in_conv1 = conv1.in_channels * conv1.kernel_size * conv1.kernel_size
        conv1.W = np.random.randn(*conv1.W.shape) * np.sqrt(2.0 / fan_in_conv1)
        conv1.b = np.zeros_like(conv1.b)
        conv1.params['W'] = conv1.W
        conv1.params['b'] = conv1.b

        conv2 = self.layers[2]
        fan_in_conv2 = conv2.in_channels * conv2.kernel_size * conv2.kernel_size
        conv2.W = np.random.randn(*conv2.W.shape) * np.sqrt(2.0 / fan_in_conv2)
        conv2.b = np.zeros_like(conv2.b)
        conv2.params['W'] = conv2.W
        conv2.params['b'] = conv2.b

        fc1 = self.layers[4]
        fc1.W = np.random.randn(*fc1.W.shape) * np.sqrt(2.0 / fc1.W.shape[0])
        fc1.b = np.zeros_like(fc1.b)
        fc1.params['W'] = fc1.W
        fc1.params['b'] = fc1.b

        fc2 = self.layers[-1]
        fc2.W = np.random.randn(*fc2.W.shape) * np.sqrt(2.0 / fc2.W.shape[0])
        fc2.b = np.zeros_like(fc2.b)
        fc2.params['W'] = fc2.W
        fc2.params['b'] = fc2.b

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Accept either [N, 784] or [N, 1, 28, 28].
        if X.ndim == 2:
            outputs = X.reshape(X.shape[0], 1, 28, 28)
        elif X.ndim == 4:
            outputs = X
        else:
            raise ValueError("Model_CNN forward expects [N, 784] or [N, 1, 28, 28].")

        outputs = self.layers[0](outputs)
        outputs = self.layers[1](outputs)
        outputs = self.layers[2](outputs)
        outputs = self.layers[3](outputs)
        self._conv_feature_shape = outputs.shape

        outputs = outputs.reshape(outputs.shape[0], -1)
        for layer in self.layers[4:]:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers[4:]):
            grads = layer.backward(grads)
        grads = grads.reshape(self._conv_feature_shape)
        grads = self.layers[3].backward(grads)
        grads = self.layers[2].backward(grads)
        grads = self.layers[1].backward(grads)
        grads = self.layers[0].backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)

        self.conv_out_channels = params['config']['conv_out_channels']
        self.conv_kernel_size = params['config']['conv_kernel_size']
        self.conv_stride = params['config']['conv_stride']
        self.conv_padding = params['config']['conv_padding']
        self.conv2_out_channels = params['config']['conv2_out_channels']
        self.conv2_kernel_size = params['config']['conv2_kernel_size']
        self.conv2_stride = params['config']['conv2_stride']
        self.conv2_padding = params['config']['conv2_padding']
        self.hidden_dim = params['config']['hidden_dim']
        self.num_classes = params['config']['num_classes']
        self.dropout_p = float(params['config'].get('dropout_p', 0.0))

        out_h = (28 + 2 * self.conv_padding - self.conv_kernel_size) // self.conv_stride + 1
        out_w = out_h
        out_h2 = (out_h + 2 * self.conv2_padding - self.conv2_kernel_size) // self.conv2_stride + 1
        out_w2 = (out_w + 2 * self.conv2_padding - self.conv2_kernel_size) // self.conv2_stride + 1
        conv_feat_dim = self.conv2_out_channels * out_h2 * out_w2

        self.layers = [
            conv2D(
                in_channels=1,
                out_channels=self.conv_out_channels,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
            ),
            ReLU(),
            conv2D(
                in_channels=self.conv_out_channels,
                out_channels=self.conv2_out_channels,
                kernel_size=self.conv2_kernel_size,
                stride=self.conv2_stride,
                padding=self.conv2_padding,
            ),
            ReLU(),
            Linear(in_dim=conv_feat_dim, out_dim=self.hidden_dim),
            ReLU(),
        ]
        if self.dropout_p > 0:
            self.layers.append(Dropout(self.dropout_p))
        self.layers.append(Linear(in_dim=self.hidden_dim, out_dim=self.num_classes))
        self._conv_feature_shape = None

        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, layer_param in zip(optimizable_layers, params['layers']):
            layer.W = layer_param['W']
            layer.b = layer_param['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = layer_param['weight_decay']
            layer.weight_decay_lambda = layer_param['lambda']
        
    def save_model(self, save_path):
        params = {
            'config': {
                'conv_out_channels': self.conv_out_channels,
                'conv_kernel_size': self.conv_kernel_size,
                'conv_stride': self.conv_stride,
                'conv_padding': self.conv_padding,
                'conv2_out_channels': self.conv2_out_channels,
                'conv2_kernel_size': self.conv2_kernel_size,
                'conv2_stride': self.conv2_stride,
                'conv2_padding': self.conv2_padding,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes,
                'dropout_p': self.dropout_p,
            },
            'layers': [],
        }
        for layer in self.layers:
            if layer.optimizable:
                params['layers'].append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(params, f)