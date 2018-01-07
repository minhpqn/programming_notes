# Keras notes

Author: Pham Quang Nhat Minh

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