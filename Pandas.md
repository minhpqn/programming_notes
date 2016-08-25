# Sử dụng Pandas cho xử lý dữ liệu
==================================

Ghi chép cá nhân khi học, sử dụng Pandas cho xử lý dữ liệu.



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
