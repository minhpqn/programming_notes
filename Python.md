# Python Notes
Ghi chép cá nhân khi học và sử dụng ngôn ngữ lập trình Python.

## Duyệt các số từ N đến 1 theo thứ tự giảm dần
Yêu cầu là duyệt các số từ N đến 1 theo thứ tự N, N-1, N-2,...,1.

- Cách 1: Dùng class xrange với step = -1
```for i in xrange(N,0,-1):
       print i
```
- Cách 2: Đảo ngược của dãy
```for i in reversed(range(1,N+1)):
       print i
```

## Convert 1 số sang số tự nhiên


## Xử lý từng dòng của của file csv từ stdin
Vấn đề: xử lý từng dòng theo định dạng csv từ stdin để có thể thực hiện chương trình theo kiểu pipeline.

Ví dụ:
    cat csv_file | process_csv_file.py

Giải pháp: Convert stream từ stdin sử dụng hàm iter:
    iter(sys.stdin.readline, '')

Xem thêm tại: [How to read a CSV file from a stream and process each line as it is written?](http://stackoverflow.com/questions/6556078/how-to-read-a-csv-file-from-a-stream-and-process-each-line-as-it-is-written) trên stackoverflow.com

## Cài đặt python module
- [https://docs.python.org/2/install/](https://docs.python.org/2/install/).
- [http://python-packaging-user-guide.readthedocs.org/en/latest/installing/#installing-from-pypi](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/#installing-from-pypi).


