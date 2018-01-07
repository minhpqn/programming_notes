# Ghi chép khi sử dụng Pytorch cho Deep Learning

Created by: Pham Quang Nhat Minh

Date created: December 21, 2017

## InTensor

Integer Tensor

## Hàm view

The view function is meant to reshape the tensor. 

Reference: [https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch](https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch)

torch.nn only supports mini-batches The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.

For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.

## Mutation operator

Any operation that mutates a tensor in-place is post-fixed with an _ For example: x.copy_(y), x.t_(), will change x.

## Chuyển đổi giữa Tensor và numpy

Việc chuyển đổi giữa Tensor và numpy không thực sự ý nghĩa. Hai kiểu chia sẻ
vùng bộ nhớ nên thay đổi một giá trị một biến sẽ dẫn tới sự thay đổi của biến kia.

```
import torch
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)
```

## Definition of define-by-run

Your backprop is defined by how your code is run, and that every single iteration can be different.

## Pytorch only support mini-batch

torch.nn only supports mini-batches The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.

For example, ```nn.Conv2d``` will take in a 4D Tensor of ```nSamples x nChannels x Height x Width```.

If you have a single sample, just use ```input.unsqueeze(0)``` to add a fake batch dimension.
