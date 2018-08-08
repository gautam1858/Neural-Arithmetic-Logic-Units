import argparse
import numpy as np
import mxnet as mx
import warnings

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
from mxnet.gluon.data import vision

warnings.simplefilter(action='ignore', category=DeprecationWarning)

inputs, hiddens, outputs = 784, 200, 10
learning_rate = 0.01
epochs = 100
batch_size = 20

ctx = mx.cpu()

class NALU(Block):
    def __init__(self, **kwargs):
        super(NALU, self).__init__(**kwargs)
        
    def forward(self, X):
        W_hat = mx.nd.random.normal(0, 0.01)
        M_hat = mx.nd.random.normal(0, 0.01)
        G = mx.nd.random.normal(0, 1)
        W1 = mx.nd.Activation(data=W_hat, act_type='tanh')  
        W2 = mx.nd.Activation(data=M_hat, act_type='sigmoid') 
        W  = W1*W2
        a = mx.nd.multiply(X,W)
        g = mx.nd.multiply(X,G)
        g = mx.nd.Activation(data=g, act_type='sigmoid')

        z1 = mx.nd.abs(X)
        z = mx.nd.log(z1 + 1e-7)
        m = mx.nd.multiply(z,W)

        y = (g*a) + (1-g)*m
        
        return y

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
    
train_data = mx.gluon.data.DataLoader(vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

def mlp():
    model = nn.Sequential()
    with model.name_scope():
        model.add(nn.Dense(hiddens, activation="sigmoid"))
        model.add(nn.Dense(outputs, activation="sigmoid"))
        model.add(NALU())
        dist = mx.init.Uniform(1/np.sqrt(float(hiddens)))
        model.collect_params().initialize(dist, ctx=ctx) 
    return model

def train():
    model = mlp()   
    loss = gluon.loss.L2Loss()
    optimizer = gluon.Trainer(model.collect_params(), 'RMSProp', {'learning_rate': learning_rate})

    for e in range(epochs):
        cumulative_error = 0
        for i, (data, labels) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1, inputs))
            labels = nd.one_hot(labels, 10, 1, 0).as_in_context(ctx)
            with autograd.record():
                output = model(data)
                error = loss(output, labels)
            error.backward()
            optimizer.step(data.shape[0])
            cumulative_error += nd.sum(error).asscalar()
        print("Epoch [%d/%d]: error: %.4f" % (e+1, epochs, cumulative_error/len(train_data)))    
    model.save_params("mxnet.model")

def predict():
    model = mlp()
    model.load_params("mxnet.model", ctx)
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(ctx).reshape((-1, inputs))
        label = label.as_in_context(ctx)
        output = model(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    print("accuracy: %.2f%%" % (acc.get()[1] * 100))

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train' )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.action == "predict":
        predict()
    if FLAGS.action == "train":
        train()
