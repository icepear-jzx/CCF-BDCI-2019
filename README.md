# CCF-BDCI-Car-Sales-Forecast

## fullyconnected.py
*update 11/9*
### description
This model adds several fully connected nerual layers directly to the input. Here are some features of the model:
- Trained with MSE error and l2 normalization.
- Using drop out in every hidden layer.
- Using glorot to initialize the weights.
### usage
```shell
python fullyconnected.py
```
### results
This model performs badly. It has a big error and converges slowly and unstably. The score it gets is usually below 0.1.

## fm.py
*update 11/9*
### description
This is a very simple implementation of **Factorization Machine** model, a very useful and powerful method to deal with one-hot vectors. It's wildly used in recommendation systems. Check [this](https://blog.csdn.net/google19890102/article/details/45532745) for more details.
### usage
```shell
python fm.py
```
### results
Much to my suprise, **such a simple vesion of FM model gains great accuracy and speed. There aren't any deep learning methods in this model, and it can easily get a score over 0.5.**

## pre-requirements
```shell
pip3 install -r requirements.txt --user
```
