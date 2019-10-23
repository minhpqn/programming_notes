# Ghi chép cá nhân khi làm NLP

Tạo bởi: Phạm Quang Nhật Minh

Ngày tạo: 16/06/2019

## Biaffine classifier nghĩa là gì


Ok, so I found an answer. What the papers meant with biaffine classifier is as follows. In neural networks, we have the usual transformation like Ax+b, where A is a matrix, or more conventionally Wx+b, where W is a weight matrix, x is an input vector and b is a bias.

So this Wx+b transformation is an affine transformation, while if we apply another transformation to this namely W(Wx+b)+b, then this is an affine transformation applied twice. Hence, biaffine.

Reference:

- [https://math.stackexchange.com/questions/2792369/what-is-a-bi-affine-classifier](https://math.stackexchange.com/questions/2792369/what-is-a-bi-affine-classifier)

## How to freeze bert model and just train a classifier?

```
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

for param in model.bert.parameters():
    param.requires_grad = False
```

Chú ý khi dùng optimizer phải có tùy chọn sau đây:

```
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
```

References:

- [https://github.com/huggingface/pytorch-pretrained-BERT/issues/400](https://github.com/huggingface/pytorch-pretrained-BERT/issues/400)
- [BERT tuning all parameters?](https://github.com/huggingface/pytorch-pretrained-BERT/issues/252)
- [PyTorch example: freezing a part of the net (including fine-tuning)](https://gist.github.com/L0SG/2f6d81e4ad119c4f798ab81fa8d62d3f)
- [How to confirm freezing is working?](https://discuss.pytorch.org/t/how-to-confirm-freezing-is-working/22648)


## Cách sử dụng multi-bleu.perl cho đánh giá kết quả dịch máy

```
./multi-bleu.perl reference < output
```

Có thể sử dụng [sacreBLEU](https://github.com/mjpost/sacreBLEU) để tính BLEU score với output đã được detokenize.

```
cat output | sacrebleu reference
```

## Dùng hidden state ở vị trí nào để biểu diễn câu với BERT

Q: Why not use the hidden state of the first token as default strategy, i.e. the [CLS]?

A: Because a pre-trained model is not fine-tuned on any downstream tasks yet. In this case, the hidden state of [CLS] is not a good sentence representation. If later you fine-tune the model, you may use [CLS] as well.

Reference: [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

Q: Why not the last hidden layer? Why second-to-last?

A: The last layer is too closed to the target functions (i.e. masked language model and next sentence prediction) during pre-training, therefore may be biased to those targets. If you question about this argument and want to use the last hidden layer anyway, please feel free to set pooling_layer=-1.

## Biểu diễn câu khi không fine-tune BERT

-1 means the last hidden layer. There are 12 or 24 hidden layers, so -1,-2,-3,-4 means 12,11,10,9 (for BERT-Base.) It's extracted for each token. There is not any "sentence embedding" in BERT (the hidden state of the first token is not a good sentence representation). If you want sentence representation that you don't want to train, your best bet would just to be to average all the final hidden layers of all of the tokens in the sentence (or second-to-last hidden layers, i.e., -2, would be better).

If you're using the latest version of the repo then you don't need to tokenize it yourself, the Chinese character tokenization is handled by tokenization.py.

Reference: [Features extracted from layer -1 represent sentence embedding for a sentence? #71](https://github.com/google-research/bert/issues/71)


