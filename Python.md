# Python Notes

Ghi chép cá nhân khi học và sử dụng ngôn ngữ lập trình Python.

Tác giả: Phạm Quang Nhật Minh

## Create a DataFrame from a list of list

```
import pandas as pd
a_list = [ ["a1", "b1"], ["a2", "b2"]]
headers = ["a", "b"]
df = pd.DataFrame(a_list, columns=headers)
```

## Filter rows with NULL value in a certain column

```
import pandas as pd
df = df[pd.notnull(df['Number'])]
```

## Generate random data frame

```
import pandas as pd
import numpy as np
x = 2* np.random.randn(100) + 1
y = 2 * x + 0.5
df = pd.DataFrame(data={"x": x, "y":y})
```

## Read an excel file in a specific sheet

```
import pandas as pd
df = pd.read_excel(path_to_file, sheetname = 'Hub ID related')
```

## Replace columns' values with conditions

```
df.ix[df["Intent"] == "reset_pass", "Intent"] = "unable_login"
```

## Read excel files

Dùng function pandas.read_excel()

## Select by Boolean indexes

Sử dụng hàm ```loc```.

## Select some specific columns in a data frame using indexes

Use ```iloc```.

```
df.iloc[:,2]
df.iloc[:,1:3]
```

## CheatSheet: Data Exploration using Pandas in Python

Tham khảo tại [đây](https://www.analyticsvidhya.com/blog/2015/07/11-steps-perform-data-analysis-pandas-python).

## Thay đổi giá trị NaN trong pandas

Fill các giá trị Null cho một số columns.

```
cols = ['Suddenly_breaking_Flag', 'Harsh_acceleration_Flag',
        'Quick_Changes_in_right', 'Quick_Changes_in_left', 'Shock_Flag',
        'bCall_Flag', 'eCall_Flag']
for col in cols:
    df.ix[pd.isnull(df[col]), col] = 0
```

## Bỏ các cột trong pandas

Sử dụng hàm ```drop```.

```
X = df.drop(['label', 'label2', 'metadata_file'], axis = 1)
```

Tham khảo tại [đây](http://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas).

## Tham khảo 12 kỹ thuật xử lý dữ liệu với pandas

[https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/](https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/)

## Chuyển giá trị nhãn thành các mức (levels)

Tham khảo về categorical data của pandas.

[http://pandas.pydata.org/pandas-docs/stable/categorical.html](http://pandas.pydata.org/pandas-docs/stable/categorical.html)

```
# Chuyển một cột thành dạng category
df['label'] = df['label'].astype('category')
# Gán giá trị cho cột mới theo các code của cột label
df['label2'] = df['label'].cat.codes
```

## Chọn cột trong DataFrame

```
import pandas as pd
# cột thứ nhất
df.ix[:,0]
```

## Ghi nội dung của DataFrame ra file HTML

```
# Cách 1:
with open('my_file.html', 'w') as fo:
    fo.write(df.to_html())
# Cách 2:
df.to_html(open('my_file.html', 'w'))
# Cách 3:
with open('my_file.html', 'w') as fo:
    tsod.to_html(fo)
```

Vì ở chế độ mặc định, nếu nội dung trong một cell có độ dài vượt quá 50, pandas sẽ thay thế phần vượt quá bằng các dấu chấm, nên muốn hiển thị toàn bộ nội dung của các cell, ta phải dùng tuỳ chọn sau đây. Tham khảo: [http://pandas.pydata.org/pandas-docs/stable/options.html](http://pandas.pydata.org/pandas-docs/stable/options.html).

```
pd.set_option('display.max_colwidth', -1)
```

Tham khảo:

- [pandas.DataFrame.to_html()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_html.html)
- [Save pandas to_html as a file](http://stackoverflow.com/questions/14897833/save-pandas-to-html-as-a-file), trên stackoverflow.

## Sampling dữ liệu từ DataFrame

```
import pandas as pd
import numpy as np

indexes = np.random.choice(range(0, df.shape[0]), 
                           size=num_sample,                        
                           replace=True)

sample = df.iloc[sorted(indexes)])
```

## Tài liệu tham khảo về Pandas

- 10 Minutes to pandas. Bài hướng dẫn căn bản về cách sử dụng Pandas: [http://pandas.pydata.org/pandas-docs/stable/10min.html](http://pandas.pydata.org/pandas-docs/stable/10min.html).
- Pandas Cookbook. Xem tại link: [http://pandas.pydata.org/pandas-docs/stable/cookbook.html](http://pandas.pydata.org/pandas-docs/stable/cookbook.html).
- Pandas API Reference. Xem tại đây: [http://pandas.pydata.org/pandas-docs/stable/api.html](http://pandas.pydata.org/pandas-docs/stable/api.html)

## Kết nối các DataFrame dựa vào khoá chung

Tham khảo: [http://pandas.pydata.org/pandas-docs/stable/merging.html](http://pandas.pydata.org/pandas-docs/stable/merging.html)

```
merge(left, right, how='inner', on=None, left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False)
```

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


## Use left-justify with StyleFrame

```
from StyleFrame import StyleFrame, Styler, utils
StyleFrame(df2, styler_obj=Styler(horizontal_alignment=utils.horizontal_alignments.left, vertical_alignment=utils.vertical_alignments.top)).to_excel(filepath).save()
```

## Write data frame to excel with style

Sử dụng StyleFrame

```
from StyleFrame import StyleFrame
StyleFrame(df).to_excel(path_to_xlsx_output).save()
```

## Sorting with custom comparision in Python 3

Python 3 đã bỏ đối số `cmp=` trong hàm sort của Python 2. Tuy nhiên chúng ta vẫn có thể cài đặt hàm sort thực hiện hàm comparision tùy biến với wrapper sau đây.

```
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def reverse_numeric(x, y):
  return y - x

sorted([5, 2, 4, 1, 3], key=cmp_to_key(reverse_numeric))
```

Reference: [https://docs.python.org/3/howto/sorting.html](https://docs.python.org/3/howto/sorting.html)

## Call super class's method

```
super(CNN, self).__init__(config)
```

## Load data with pickle

Python 3

```
import pickle
pairs = pickle.load(open("./file_path.pkl", "rb"))
```

## Update conda

```
conda update -n base conda
```

## Hàm unzip

```
a = [(1,2),(3,4),(5,6)]
b,c = zip(*a)
print(b) # (1, 3, 5)
print(c) # (2, 4, 6)
```

## Biến \_\_all\_\_

Danh sách các public objects khai báo bởi ```import *```

Tham khảo: [https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python](https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python)

Ví dụ [https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/datasets/__init__.py](https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/datasets/__init__.py)

```
from .text import LinedTextDataset
from .open_subtitles import OpenSubtitles2016
from .wmt import WMT16_de_en
from .multi_language import MultiLanguageDataset
from .coco_caption import CocoCaptions
__all__ = ('LinedTextDataset',
           'OpenSubtitles2016',
           'MultiLanguageDataset',
           'WMT16_de_en',
           'CocoCaptions')
```

## Hàm getattr

```
getattr(object, name[, default]) -> value
```
    
Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y. When a default argument is given, it is returned when the attribute doesn't exist; without it, an exception is raised in that case.

## Use pathlib to make directory

```
import pathlib
pathlib.Path("dir_name").mkdir(exist_ok=True, parents=True)
```

## Sử dụng \_\_init\_\_.py

```
from .intent import IntentClassifier
```

## Defaultdict of defaultdict

```
defaultdict(lambda : defaultdict(int))
```

Refer [https://stackoverflow.com/questions/5029934/python-defaultdict-of-defaultdict](https://stackoverflow.com/questions/5029934/python-defaultdict-of-defaultdict)

## Scan all matches of a regular expression

Use ```findall```


## Read the json file, keep the order

```
json.load(fi, object_pairs_hook=collections.OrderedDict)
```

## Write unit test in python

```
import unittest
class ABCTest(unittest.TestCase):
    def testA(self):
        # write test here
        # self.assertEqual()
        pass
    def testB(self):
        # write test here
        pass
```

## Write json data to a file

```
with open(file_name, 'w') as fo:
    json.dump(json_data, fo)
```

## Call Resful API in python3

```
import requests
r = requests.get(base_url, params={'q': sentence})
```

## Export environment file yml

```
conda-env export --name root > environment.yml
```

## Read json data file into a dictionary object

```
import json
with open(filepath) as f:
    data = json.load(f)
```

## Random sample from a list

```
import random
random.sample([1,2,3,4,5,6], k=3)
```

## Quy ước đặt tên biến khi viết python class

Sử dụng tên biến dạng ```self.w_``` cho những biến được tạo ra khi gọi một method
khác (thay vì khởi tạo ở hàm khởi tạo).

## Using argparse for options

```
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

## Add an element in a certain position of a python list

```
a = [1, 2, 3, 4]
a.insert(1,5)
>> [1, 5, 2, 3, 4]
```

## Uninstall a package with conda

```
conda uninstall keras
```

## Replace multiple spaces with a single space in a string

```
import re
astring = 'I am a   student'
re.sub(r'\s{2,}', ' ', astring)
```

## Read MySQL database with Python

Sử dụng PyMySQL: [https://github.com/PyMySQL/PyMySQL](https://github.com/PyMySQL/PyMySQL)

Cài đặt PyMySQL bằng pip

```
pip install PyMySQL
```

Kết nối tới mysql database

```
import pymysql
connection = pymysql.connect(host='localhost',
                             user=username, 
                             password=password,
                             db=dbname,
                             charset='utf8mb4')
```

Sử dụng cursor để lấy kết quả

```
with connection.cursor() as cursor:
    sql = "SELECT * FROM `mydb`"
    cursor.execute(sql)
    result = cursor.fetchall()
```

Xem thêm tài liệu về PyMySQL tại: [http://pymysql.readthedocs.io/en/latest/index.html](http://pymysql.readthedocs.io/en/latest/index.html)

## Python Exception Handling Techniques

Reference: [https://doughellmann.com/blog/2009/06/19/python-exception-handling-techniques/](https://doughellmann.com/blog/2009/06/19/python-exception-handling-techniques/)

## Unzip a list of tuples

Use ```zip(*list)```

```
>>> l = [(1,2), (3,4), (8,9)]
>>> zip(*l)
[(1, 3, 8), (2, 4, 9)]
```

## Randomly get N elements from a list

```
import numpy
a = [1, 2, 3, 4, 5]
# with replacement
numpy.random.choice(a, 2, replace=True)
# without replacement
numpy.random.choice(a, 2, replace=False)
```

## Get group in regular expression match

```
import re
r = re.search(r'', s)
r.group(1)
```

## Pop the last element of a python list

```
a = [1, 2, 3]
a.pop() # [1, 2]
```

## Set pythonpath before import statements

```
import sys
sys.path.append(path_to_lib)
```

## Kiểm tra version của keras

```
python -c "import keras; print( keras.__version__ )"
```

## Cài đặt package trong pytorch

```
conda install torchtext -c soumith
```

## Create a separate environment

```
conda create --name bunnies python=3 astroid babel
```

## How to use \*args and \*\*kwargs in Python

\*kwargs

```
def test_var_kwargs(farg, **kwargs):
    print "formal arg:", farg
    for key in kwargs:
        print "another keyword arg: %s: %s" % (key, kwargs[key])
test_var_kwargs(farg=1, myarg2="two", myarg3=3)
```

\*args

```
def test_var_args(farg, *args):
    print "formal arg:", farg
    for arg in args:
        print "another arg:", arg
test_var_args(1, "two", 3)
```

## Delete an python environment with conda

```
conda env remove -n <name_of_env>
```

## Show python environments with conda

```
conda info --envs
```

## Python regular expression for email string

```
r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
```

## Create a list of regular expressions and match in that order

```
regexes = [
    # your regexes here
    re.compile('hi'),
#    re.compile(...),
#    re.compile(...),
#    re.compile(...),
]
mystring = 'hi'

if any(regex.match(mystring) for regex in regexes):
    print 'Some regex matched!'
```

References: [Python: Elegant way to check if at least one regex in list matches a string](http://stackoverflow.com/questions/3040716/python-elegant-way-to-check-if-at-least-one-regex-in-list-matches-a-string)

## Check if a directory exists or not and create if not existing

```
import os
dir = "./data"
if not os.path.isdir(dir):
    os.mkdir(dir)
```

## Write data frame into file csv and html (dạng table)

Sử dụng pandas.DataFrame.to_csv()

## Find out the directory of an input file

Sử dụng "os.path.dirname"

## raw_input() in python3

Đổi thành input() trong python3

## Check blank lines in python

```
import re
if re.search(r'^[\s\t]*$', line)
```

## Shuffle a list in python

Use ```random.shuffle```

## Use nltk to untag

Untag một câu trong đó mỗi từ được gắn với POS Tag. Ví dụ câu sau đây:

The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd \`\`/\`\` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.

```
from nltk.tag.util import untag
sen = 'The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd \`\`/\`\` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.'
tagged_sen = [ pair.split('/') for pair in sen.split()]
sen = untag(tagged_sen)
```

## How to match any string from a list of strings in regular expressions in python?

```
string_lst = ['fun', 'dum', 'sun', 'gum']
x="I love to have fun."
print re.findall(r"(?=("+'|'.join(string_lst)+r"))",x)
```
## Check if a directory exists or not. If it does not exist, create one.

```
import os
if not os.path.isdir(dir):
    os.makedirs(dir)
```

## Convert a multiple-sheet excel file into multiple csv files

Yêu cầu: Convert mỗi sheet trong file excel thành một file csv.

```
dfs = pd.read_csv('QuestionSet-FRT-1.xlsx', sheetname = None)
```

Kết quả trả về của hàm trên là một dictionary của các sheets trong file excel.

## Draw graph for sigmoid function

g(z) = 1/( 1 + exp(-z) )

```
# generate random number from -5 to 5
import numpy as np
from matplotlib import pyplot as plt
z = np.arange(-5., 5., 0.2)
# calculate g(z)
y = 1/(1 + np.exp(-z))
# plot g(z) using matplotlib
plt.plot(z, y)
plt.show()
```

## Change to a directory

Use ```os.chdir(path)```

## Replace all tabs with spaces in a string

```
import re
re.sub(r"\s+", " ", text)
```

## Get all content of a file

```
with open(fname) as f:
    content = f.readlines()
```

## Open a file to write

```
f = open('/tmp/spam', 'w')
f.write(S)
```

## Get basename of a file path

```
os.path.basename(path)
os.path.splitext('path_to_file')[0]
```

## Convert notebooks thành các định dạng khác nhau

```
# Chuyển sang HTML
ipython nbconvert --to html notebook.ipynb
```

Tham khảo [Converting notebooks to other formats](https://ipython.org/ipython-doc/1/interactive/nbconvert.html)

## Directory listing in Python

Sử dụng os.listdir()

```
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
```

## iPython Notebook Keyboard Shortcuts

Reference: [iPython Notebook Keyboard Shortcuts](http://johnlaudun.org/20131228-ipython-notebook-keyboard-shortcuts/).

## Try catch trong Python

```
try:
  # process here
except Exception:
  # process here
```

## Import Python module from subdirectory

Assume we need to import the module ```article.py``` in ```./lib``` Use the following.

```
import sys
sys.path.append('./lib')
```

## Xử lý file XML với Python

Ví dụ bài 54 trong "100 NLP Drill Exercises".

```
from xml.dom import minidom
xmldoc = minidom.parse('../../data/nlp.txt.xml')
token_list = xmldoc.getElementsByTagName('token')
for tknode in token_list:
    info = []
    for tag in ['word', 'lemma', 'POS']:
        node = tknode.getElementsByTagName(tag)
        info.append(node[0].firstChild.data)
    print '\t'.join(info)
```

## Đọc vào từ stdin

```
for line in sys.stdin:
    line = line.strip()
    print line
```

## Iterating Over Arrays

Tham khảo [http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html](http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html)

## Tính tổng các phần tử theo hàng/cột của ma trận với python numpy

```
x = np.array([[1001,1002],[3,4]])
# Tính tổng các phần tử trên mỗi hàng của ma trận
np.sum(x, axis=1)
>> array([2003,    7])
# Tính tổng các phần tử trên các cột của ma trận
np.sum(x, axis=0)
>> array([1004, 1006])
```

## Sự khác nhau giữa () và [] trong list comprehension

() sẽ tạo ra ```generator``` trong khi [] sẽ tạo ra một ```list```. Tuỳ theo nhu cầu sử dụng mà ta có thể chọn () hoặc [].

## Một cách vận dụng defaultdict và generator rất hay mà mình cần học tập

```
from collections import defaultdict
sentences = ['the king loves the queen', 'the queen loves the king',
             'the dwarf hates the king', 'the queen hates the dwarf',
             'the dwarf poisons the king', 'the dwarf poisons the queen']
def Vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary
def docs2bow(docs, dictionary):
    """Transforms a list of strings into a list of lists
    where each unique item is converted into a unique
    integer.
    """
    for doc in docs:
        yield [dictionary[word] for word in doc.split()]
vocabulary = Vocabulary()
sentences_bow = list(docs2bow(sentences, vocabulary))
```

Trong hàm ```docs2bow```, mỗi khi gọi ```dictionary[word]```, default_factory sẽ tự động gán giá trị cho khoá trong biến ```word``` là kích thước của ```dict``` hiện tại.


## Về defaultdict và default factory trong python

Ví dụ khi kiểu giá trị mặc định là ```list```

```
from collections import defaultdict
cities_by_state = defaultdict(list)
```

Trong Perl thì không cần một lớp riêng vì Perl sẽ tự động định kiểu khi giá trị được gán.

Tham khảo [https://www.accelebrate.com/blog/using-defaultdict-python/](https://www.accelebrate.com/blog/using-defaultdict-python/)

## Giải thích dễ hiểu về generator trong python

[Improve Your Python: 'yield' and Generators Explained](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/).

Take-away points:

- ```generators``` are used to generate a series of values
yield is like the return of generator functions
- The only other thing yield does is save the "state" of a generator function
- A generator is just a special type of iterator
- Like iterators, we can get the next value from a generator using next()
    * for gets values by calling next() implicitly

## Cài đặt một package mới với conda

Ví dụ: lệnh sau sẽ cài đặt seaborn với conda:

```
conda install seaborn
# list packages in the current environment
conda list
```

Tham khảo: [http://conda.pydata.org/docs/using/pkgs.html](http://conda.pydata.org/docs/using/pkgs.html)

## Quản lý Python version với conda

Tham khảo: [http://conda.pydata.org/docs/py2or3.html](http://conda.pydata.org/docs/py2or3.html)

## Quản lý environments với conda

Tham khảo tại: [http://conda.pydata.org/docs/using/envs.html](http://conda.pydata.org/docs/using/envs.html)

## Tài liệu tham khảo: numpy cho matlab users

- [NumPy for MATLAB users](http://mathesaurus.sourceforge.net/matlab-numpy.html)
- [Numpy for Matlab users trên docs.scipy.org)](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html)

## Câu lệnh tương đương với lệnh find() trong matlab

```
a = np.array([1, 3, -1, 0.5, 0.3, 10])
(a > 0.5).nonzero()
Out[26]: (array([0, 1, 5]),)
a[ (a > 0.5).nonzero() ]
Out[27]: array([  1.,   3.,  10.])
```

## Sắp xếp mảng, trả về indices ban đầu của các phần tử trong mảng đã sắp xếp

```
>>> s = [2, 3, 1, 4, 5]
>>> sorted(range(len(s)), key=lambda k: s[k])
[2, 0, 1, 3, 4]
>>>
```

Với Python numpy ta có thể dùng hàm ```numpy.argsort```

## Split tại ký tự phân tách đầu tiên

Dùng hàm ```str.split([sep[,maxsplit]])```

```
str.split(' ', 1)
```

## Dùng *args và **kargs để unpack variables khi định nghĩa hàm

Xem thêm [http://agiliq.com/blog/2012/06/understanding-args-and-kwargs/](http://agiliq.com/blog/2012/06/understanding-args-and-kwargs/)

## Logical Functions trong python numpy

Tham khảo [http://docs.scipy.org/doc/numpy/reference/routines.logic.html](http://docs.scipy.org/doc/numpy/reference/routines.logic.html)

## Tính pdf của multivariate Gaussian Distribution

Dùng thư viện (Tham khảo: [http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html))

```
from scipy.stats import multivariate_normal
p = multivariate_normal.pdf(X, mean=mu, cov=Sigma2)
```

## Tìm indices của các phần tử thoả mãn điều kiện cho trước trong một mảng

```
# use list comprehension
[i for i in xrange(len(arr)) if sastified(arr[i]) ]
```

## Về cú pháp numpy.r_

[Understanding the syntax of numpy.r_() concatenation](http://stackoverflow.com/questions/14468158/understanding-the-syntax-of-numpy-r-concatenation)

'n,m' tells r_ to concatenate along axis=n, and produce a shape with at least m dimensions:

```
In [28]: np.r_['0,2', [1,2,3], [4,5,6]]
Out[28]:
array([[1, 2, 3],
       [4, 5, 6]])
```

Xem thêm [http://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.r\_.html](http://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.r_.html)

## Thao tác Selection/Slice trong numpy.ndarray

```
import numpy as np
a = np.array( range(1,11) ).reshape(2,5)
a[:, 2:]
```

## Truy cập một cột của numpy array

```
import numpy as np
a = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
# Truy cập cột thứ 2 của a
a[:,[2]]
```

## Tính tổng theo các cột của numpy array

```
a = np.array([[1, 2], [3,4], [5,6]])
np.sum(a, axis=1)
Out[137]: array([ 3,  7, 11])
```

## Tài liệu tham khảo về ipython

- [IPython's Rich Display System](http://nbviewer.jupyter.org/github/ipython/ipython/blob/1.x/examples/notebooks/Part%205%20-%20Rich%20Display%20System.ipynb)
- [Ipython Notebooks Examples](http://nbviewer.jupyter.org/github/ipython/ipython/tree/1.x/examples/notebooks/)

## Rank của array trong Python numpy
Là số chiều của array.

```
import numpy as np
a = np.array([1, 2, 3])  # Create a rank 1 array
print type(a), a.shape, a[0], a[1], a[2]
a[0] = 5                 # Change an element of the array
print a   
```

## Kỹ thuật broadcasting

Được dùng khi thực hiện phép tính giữa hai mảng: 1 mảng với kích thước lớn và một mảng khác với kích thước nhỏ hơn.

```
import numpy as np
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y
```

## Chú ý khi slice một mảng trong numpy
Thay đổi giá trị của sliced array sẽ thay đổi giá trị của mảng gốc.

```
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print a[0, 1]  
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print a[0, 1]
```

## Tạo mảng với các số ngẫu nhiên trong Python numpy

```
import numpy as np

e = np.random.random((2,2)) # Create an array filled with random values
print e
```

## Set comprehension
Tương tự như dictionary comprehension, ta có thể làm như sau:

```
from math import sqrt
print {int(sqrt(x)) for x in range(30)}
```

## Dictionary comprehension

Rất tiện khi tạo dictionary.

```
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square
```

## Truy cập key, value khi duyệt dictionary
Sử dụng hàm iteritems.

```
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)
```

## Nên sử dụng list comprehension nhiều hơn thay vì sử dụng map, filter, lambda

```
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares
```

List comprehension với cấu trúc điều kiện.

```
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares
```

## Duyệt mảng đồng thời truy cập index của các phần tử

```
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
```

## Tìm kiếm các lệnh đã thực hiện trong history của ipython

```
[l for l in  _ih if l.startswith('plot')]
```

## Tài liệu tham khảo về phong cách viết code trong Python

- [The Elements of Python Style](https://github.com/amontalenti/elements-of-python-style), by amontalenti.
- [PEP 20 -- The Zen of Python](https://www.python.org/dev/peps/pep-0020/).
- [Flake8](https://flake8.readthedocs.org/en/latest/).
- [PEP 0008 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
- [Transforming Code into Beautiful, Idiomatic Python](https://www.youtube.com/watch?v=OSGv2VnC0go), Youtube video by Raymond Hettinger.

## Khắc phục lỗi IOError: [Errno 32] Broken pipe
Khi sử dụng pipeline với python trên Linux ta hay gặp lỗi đó. Ví dụ trích ra các dòng đầu của đầu ra của một chương trình Python. Để khắc phục ta thêm vào khai báo sau đây:

```
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
```
Tham khảo tại [http://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-python](http://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-python).

## Câu lệnh tương tự với grep trong Perl
Muốn chọn ra các phần tử trong danh sách (list) thoả mãn một tiêu chí nào đó thì dùng lệnh nào?

Sử dụng hàm filter. Xem ví dụ sau đây:

```
>>> fib = [0,1,1,2,3,5,8,13,21,34,55]
>>> result = filter(lambda x: x % 2, fib)
>>> print result
[1, 1, 3, 5, 13, 21, 55]
>>> result = filter(lambda x: x % 2 == 0, fib)
>>> print result
[0, 2, 8, 34]
>>>
```

## Vẽ đồ thị dạng thanh với Python (Bar chart)
Tham khảo: [http://matplotlib.org/examples/pylab_examples/barchart_demo.html](http://matplotlib.org/examples/pylab_examples/barchart_demo.html)

## Chuyển một xâu ký tự sang dạng mã hoá utf-8
```
a_string='Xâu ký tự'
utf8_string = unicode(a_string, 'utf-8')
```

Tham khảo: [python: how to convert a string to utf-8](http://stackoverflow.com/questions/4182603/python-how-to-convert-a-string-to-utf-8)

## Sắp xếp các khoá trong một hash map
Sắp xếp theo khoá:
```
sorted( a.keys() )
```

Sắp xếp theo giá trị: Giả sử có một đối tượng kiểu dict, các giá trị là các giá trị số thực. Ta muốn sắp xếp các khoá theo thứ giảm dần của các giá trị. Giải pháp là dùng hàm sorted với tuỳ chọn key=

```
a = {}
a[1] = 3
a[2] = 6
a[3] = 4
a[4] = 8
sorted( a.keys(), key = lambda x: a[x], reverse = True )
```

## Array sclice trong python
Sử dụng ```list[start_index:end_index]```. Chú ý là mảng thu được không bao gồm phần tử ở vị trí end_index.

Tham khảo: [https://developers.google.com/edu/python/lists](https://developers.google.com/edu/python/lists).

## Toán tử lambda và hàm map trong python
Vấn đề: muốn map các phần tử trong mảng bằng một hàm số trả về giá trị mới cho mỗi phần tử của mảng.

Tham khảo: [http://www.python-course.eu/lambda.php](http://www.python-course.eu/lambda.php).

## In một dòng không có ký tự xuống dòng

Yêu cầu: In ra một dòng không có ký tự xuống dòng.
```sys.stdout.write('Xâu ký tự')```

## Đối sánh xâu sử dụng regular expressions
Dùng module re trong Python
Tham khảo [https://developers.google.com/edu/python/regular-expressions](https://developers.google.com/edu/python/regular-expressions).

```
import re
re.search(r'iii', 'piiig')
```

## Mở 1 file để đọc trong Python
Tham khảo [https://developers.google.com/edu/python/dict-files](https://developers.google.com/edu/python/dict-files).

```
f = open('foo.txt', 'rU')
for line in f:
    print line
f.close()
```

Trong đó 'r' có nghĩa là mở file để đọc, 'U' là tuỳ chọn để chuyển đổi các dấu kết thức dòng thành '\n'.

## Kiểm tra xem 1 đường dẫn có phải là 1 thư mục hay không?

Sử dụng os.path.isdir(filename) trong module os của Python. Kiểm tra 1 đường dẫn có phải là 1 file hay không thì sử dụng os.path.isfile(filename).

## Xử lý đầu vào từ dòng lệnh trong Python

Mảng lưu trữ đầu vào từ dòng lệnh trong Python. Tham khảo tại đây: [http://www.diveintopython.net/scripts_and_streams/command_line_arguments.html](http://www.diveintopython.net/scripts_and_streams/command_line_arguments.html)

Python sẽ lưu trữ danh sách các đối dòng lệnh trong mảng ```sys.argv```. Lưu ý là danh sách này bao gồm cả tên của python script.

## Duyệt các số từ N đến 1 theo thứ tự giảm dần

Yêu cầu là duyệt các số từ N đến 1 theo thứ tự N, N-1, N-2,...,1.

```
- Cách 1: Dùng class xrange với step = -1
for i in xrange(N,0,-1):
       print i
- Cách 2: Đảo ngược của dãy
for i in reversed(range(1,N+1)):
       print i
```

## Convert 1 số sang số tự nhiên
Sử dụng hàm ```int```

## Xử lý từng dòng của của file csv từ stdin
Vấn đề: xử lý từng dòng theo định dạng csv từ stdin để có thể thực hiện chương trình theo kiểu pipeline.

Ví dụ:
    cat csv_file | process_csv_file.py

Giải pháp: Convert stream từ stdin sử dụng hàm iter
    iter(sys.stdin.readline, '')

Xem thêm tại: [How to read a CSV file from a stream and process each line as it is written?](http://stackoverflow.com/questions/6556078/how-to-read-a-csv-file-from-a-stream-and-process-each-line-as-it-is-written) trên stackoverflow.com

## Cài đặt python module
- [https://docs.python.org/2/install/](https://docs.python.org/2/install/).
- [http://python-packaging-user-guide.readthedocs.org/en/latest/installing/#installing-from-pypi](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/#installing-from-pypi).
