# Some notes about using Linux/Unix

Tác giả: Phạm Quang Nhật Minh

## Chạy Cabocha với loại từ điển khác

```
cabocha -P IPA
```

## Set đường dẫn tới các file .h cần thiết khi biên dịch

```
export CPPFLAGS=-I/usr/local/include
```

## Tạo symbolic link

```
ln -s {target-filename} {symbolic-filename}
```

Tham khảo: [https://www.cyberciti.biz/faq/creating-soft-link-or-symbolic-link/](https://www.cyberciti.biz/faq/creating-soft-link-or-symbolic-link/)

## sqlite3

macOS provides an older sqlite3.

Generally there are no consequences of this for you. If you build your
own software and it requires this formula, you'll need to add to your
build variables:

LDFLAGS:  -L/usr/local/opt/sqlite/lib
CPPFLAGS: -I/usr/local/opt/sqlite/include
PKG_CONFIG_PATH: /usr/local/opt/sqlite/lib/pkgconfig

## Create a new user account on *NIX 

Tham khảo

- [https://www.cyberciti.biz/faq/unix-create-user-account/](https://www.cyberciti.biz/faq/unix-create-user-account/)
- [Unix - User Administration](https://www.tutorialspoint.com/unix/unix-user-administration.htm)

## Convert markdown file into pdf file

    pandoc MANUAL.txt --latex-engine=xelatex -o example13.pdf
    pandoc -s -o doc.pdf doc.md

Tham khảo: [http://pandoc.org/demos.html](http://pandoc.org/demos.html)

## Copy a list of files in a directory

Bài toán: Copy một list các files vào một directory.

```
a=(*)
cp -- "${a[@]: -4}" ~/
```

## Convert all tabs to spaces in a file

```
cat file | tr '\' ' '
```

## Redirect all results to stdout and stderr

```
command &> file_name
```

## Show fields in a .csv of a row one per line

Use the command ```tr``` to replace ```,``` by newline character.

```
head -n 1 data/raw/IASLOG_20160613_WEB_000-WebProxyLog.csv | tr ',' '\n' | more -N
```

```more -N``` shows the line numbers.

## 8 Linux TR Command Examples

See [www.thegeekstuff.com/2012/12/linux-tr-command/](www.thegeekstuff.com/2012/12/linux-tr-command/)



