# Python Notes
Ghi chép cá nhân khi học và sử dụng ngôn ngữ lập trình Python.

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


