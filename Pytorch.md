# Ghi chép khi sử dụng Pytorch cho Deep Learning

Tạo bởi: Pham Quang Nhat Minh

Ngày tạo: December 21, 2017

## Freeze BERT layers for BertPretrainedModel

```
model = BertForSequenceClassification.from_pretrained("bert-base-uncase")
for param in model.parameters():
    param.requires_grad = False
```

## Average vector với bool masking

Input:

- Ma trận kích A thước B x L x D. Trong đó B là batch size, L là độ dài chuỗi và D là số chiều của word vector
- Ma trận mask đánh dấu các vị trí trong chuỗi, kích thước B x L

Output:

- Ma trận kích thước B x D, trong đó mỗi vector hàng là mean vector trong của các word vectors trong A

```
# x is the input tensor
# mask in the Boolean mask tensor
for i in range(x.shape[0]):
  mean_v[i] = torch.mean(x[i][mask[i]], dim=0)
```

## Check cuda version using torch

```
import torch
torch.version.cuda
```

## Check if cuda available using torch

```
In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'GeForce GTX 950M'

In [6]: torch.cuda.is_available()
Out[6]: True
```

Reference: [https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu](https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu)

## Bổ sung tokens đặc biệt trong tokenizer của transformers

```
tokenizer.add_tokens(['[E1]', [E2]])
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
# Tăng kích thước của embeddings của model
model.resize_token_embeddings(len(tokenizer))
```

## Using GPU efficiently

"Keep in mind that it is expensive to move data back and forth from the GPU. Therefore, the typical procedure involves doing many of the parallelizable computations on the GPU and then transferring just the final result back to the CPU. This will allow you to fully utilize the GPUs."


## Handy function to show properties of a tensor

```
def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))
```

With option `print_value`

```
def describe(x, print_value=True):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    if print_value:
        print("Values: \n{}".format(x))
```

## Convert TensorFlow checkpoint for BERT to Pytorch saved file

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

## model.train() và model.eval()

Khi trong mô hình có sử dụng dropout thì phải gọi lệnh `model.eval()` trước khi đưa ra dự đoán.

`model.eval()` tương đương với `model.train(False)`.

Tham khảo: 

- [https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/](https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/)

## Đọc model từ file

```
import torch
model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
```

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
