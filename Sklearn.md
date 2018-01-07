# Scikit-learn Notes
===========================

## Split data into 5 subsets

```
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
```

## model_selection in scikit-learn version 0.18

Từ version 0.20, module này sẽ bị remove.

/Users/minhpham/anaconda/lib/python3.5/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)

## Split data into training & test data

Use the function ```train_test_split``` in ```model_selection```.

Reference: [http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                      X, y, test_size=0.33, random_state=42)
```

## References for ensemble methods with scikit-learn

* [Kaggle Ensembling Guide](http://mlwave.com/kaggle-ensembling-guide/) on MLWave.
* [StackingClassifier](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/), by Rasbt.
* [MLXtend](https://github.com/rasbt/mlxtend)
* [Implementing a Weighted Majority Rule Ensemble Classifier](http://sebastianraschka.com/Articles/2014_ensemble_classifier.html), by Sesbastian Raschka.

## Feature Selection trong scikit-learn

```
# Sử dụng feature selection trong Pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
            ('feature_selection',
             VarianceThreshold(threshold=(.99 * (1 - .99)))),
            ('classification', LinearSVC())
            ])
```

## Về Model selection trong scikit-learn

Xem: [http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html](http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html)

## Đánh giá các estimator với cross-validation

Tham khảo: [http://scikit-learn.org/stable/modules/cross_validation.html](http://scikit-learn.org/stable/modules/cross_validation.html)

## Phương pháp Ensemble trong Scikit-learn

Tham khảo tại [http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html).

Trong các phương pháp [Gradient Tree Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) là một phương pháp rất hiệu quả.

Tham khảo thêm trên machinelearningmastery Ensemble Machine Learning Algorithms in Python with scikit-learn](http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)

## Scikit-learn cho sentiment analysis (1)

Bài toán: làm thế nào đọc dữ liệu sentiment analysis một cách hiệu quả. File dữ liệu gồm các câu, mỗi câu trên một dòng, cùng với nhãn của câu đó (+1: positive, -1: negative). Ví dụ:

```
+1 the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . 
+1 the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .
```

Cách đơn giản là đọc nội dung của file vào 2 biến ```sentences, y``` trong đó ```sentences``` là danh sách các câu (mỗi câu là 1 xâu ký tự) và ```y``` là danh sách các nhãn (kiểu ```np.ndarray```).

## Trích xuất đặc trưng

Tham khảo trên trang: [http://scikit-learn.org/stable/modules/feature_extraction.html](http://scikit-learn.org/stable/modules/feature_extraction.html).

Kỹ thuật feature hashing mình vẫn chưa nắm được rõ ràng. Tham khảo thêm tại các trang sau:
- [Feature Hashing](https://en.wikipedia.org/wiki/Feature_hashing) trên Wikipedia.
- Bài báo về kỹ thuật Feature Hashing trên ICML: Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola and Josh Attenberg (2009). [Feature hashing for large scale multitask learning](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf). Proc. ICML.

## Về iterators và generators trong Python

- [http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python](http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python)
- [The Python yield keyword explained](http://pythontips.com/2013/09/29/the-python-yield-keyword-explained/)
- [http://www.bogotobogo.com/python/python_function_with_yield_keyword_is_a_generator_iterator_next.php](http://www.bogotobogo.com/python/python_function_with_yield_keyword_is_a_generator_iterator_next.php)

## Python's zip, map, and lambda

- [https://bradmontgomery.net/blog/pythons-zip-map-and-lambda/](https://bradmontgomery.net/blog/pythons-zip-map-and-lambda/)

