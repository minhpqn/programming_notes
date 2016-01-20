# Python Notes
Ghi chép cá nhân khi học và sử dụng ngôn ngữ lập trình Python.

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
sorted( a.keys(), key = lambda x:, a[x], reverse = True )
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


