# Ghi chép khi sử dụng Pytorch cho Deep Learning

Created by: Pham Quang Nhat Minh

Date created: December 21, 2017


## Option padding_idx của layer Embedding

Ví dụ:

```
emb = nn.Embedding(output_size, emb_size, padding_idx=0)
```

Khi có tùy chọn `padding_idx`, đầu ra của layer `Embedding` sẽ được pad bởi 0 khi có gặp các index bằng `padding_idx`.

Tham khảo: [https://pytorch.org/docs/0.3.1/nn.html?highlight=embedding#torch.nn.Embedding](https://pytorch.org/docs/0.3.1/nn.html?highlight=embedding#torch.nn.Embedding)


## Filter parameters that does not require gradients

`parameters = filter(lambda p: p.requires_grad, net.parameters())`

Tham khảo: 

- [Can we use pre-trained word embeddings for weight initialization in nn.Embedding?](https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222)
- [Loading Glove Vectors in Pytorch](https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb)

## LSTM layer trong pytorch 

Tham khảo: [http://pytorch.org/docs/master/nn.html#lstm](http://pytorch.org/docs/master/nn.html#lstm)

- Đầu vào của mạng LSTM: (input, (h_0, c_0))
    + input: (seq_len, batch, input_size) hoặc "a packed variable length sequence"; nếu ```batch_first=True```, batch_size sẽ ở đầu
- Đầu ra của mạng LSTM: (output, (h_n, c_n))

## Use pre-trained word embeddings in pytorch

```
# pretrained is an embedding matrix
self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
```

References:

- [Example of an embedding loader?](https://github.com/pytorch/text/issues/30)
- [pytorch-wordemb](https://github.com/iamalbert/pytorch-wordemb)
- [Sentiment Analysis with Pytorch](https://github.com/vanzytay/pytorch_sentiment_rnn)
- [Can we use pre-trained word embeddings for weight initialization in nn.Embedding?](https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222)
- [Loading Glove Vectors in Pytorch](https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb)

## Giá trị -1 trong hàm view

Khi dùng đối vào -1, chiều tương ứng sẽ được suy luận từ các chiều khác.

Ví dụ:

```
import torch
a = torch.FloatTensor([[1,2,3], [4,5,6], [7,8,9]]) # Tensor 3x3
b = a.view(1,1,-1) # Tensor 1x1x9
```

## Các lớp, hàm cần nhớ trong Pytorch

- ```torch.autograd.Variable``` cho khai báo các biến
- ```torch.Tensor``` để định nghĩa các tensor
- ```torch.optim``` cho tối ưu
- ```torch.nn.Module```

## Set random seed

```
import torch
torch.manual_seed(42)
```

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
