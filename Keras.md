# Keras notes

Author: Pham Quang Nhat Minh

### Convert labels to one-hot encoding

```
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
```

### Using custom metrics in keras

```
# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

### Adding Attention layer on top of RNN layer in keras

- [keras-attention-mechanism](https://github.com/philipperemy/keras-attention-mechanism)
- [How to Develop an Encoder-Decoder Model with Attention for Sequence-to-Sequence Prediction in Keras](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
- [How to add Attention on top of a Recurrent Layer (Text Classification)](https://github.com/keras-team/keras/issues/4962)
- [Position-based Content Attention for Time Series Forecasting with Sequence-to-sequence RNNs](https://arxiv.org/pdf/1703.10089.pdf)
- [keras-language-modeling](https://github.com/codekansas/keras-language-modeling)
- [Understanding emotions — from Keras to pyTorch](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
- [cbaziotis/Attention.py](https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d)
- [Attention in Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)

### Upgrade keras and tensorflow

```
pip install keras --upgrade
```

### How to use advanced activation layers in Keras?

```
model = Sequential()
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
model.add(Dense(64, input_dim=14, init='uniform'))
model.add(act)
```

Tham khảo:

[https://stackoverflow.com/questions/34717241/how-to-use-advanced-activation-layers-in-keras](https://stackoverflow.com/questions/34717241/how-to-use-advanced-activation-layers-in-keras)