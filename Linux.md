# Some notes about using Linux/Unix

Tác giả: Phạm Quang Nhật Minh

## Find IP address on Mac OS 

External Public IP Address

```
curl ipecho.net/plain ; echo
```

Internal IP Address

```
ifconfig | grep inet
```

Hoặc vào trang: [https://www.whatismyip.org/my-ip-address](https://www.whatismyip.org/my-ip-address)

## Keep environment variables after running a shell script

- Để lưu trữ các biến môi trường sau khi chạy script dùng lệnh
    * . my_script.sh
    * source my_script.sh

## Giải nén file .tar.bz2 trên linux

```
tar jvxf cabocha-0.69.tar.bz2
```

## Xem version của cuda

      cat /usr/local/cuda/version.txt

## Lỗi Centos warning: setlocale

Centos warning: setlocale: LC_CTYPE: cannot change locale (UTF-8): No such file or directory 

Thêm các dòng sau vào file `/etc/environment`

```
vi /etc/environment

add these lines...

LANG=en_US.utf-8
LC_ALL=en_US.utf-8
```

## Change owner of a directory

```
sudo chown -R ubuntu:ubuntu OpenNMT-py
```

## Identify CUDA version

```
cat /usr/local/cuda/version.txt
```

## Exit broken ssh session

Gõ Enter, ~, .

Tham khảo: [https://askubuntu.com/questions/29942/how-can-i-break-out-of-ssh-when-it-locks](https://askubuntu.com/questions/29942/how-can-i-break-out-of-ssh-when-it-locks)


## Add an user with home directory

```
sudo useradd -m -d /home/ubuntu/minhpham/ abc
```

Change password

```
passwd abc
```

## Install git locally on the server

Cài đặt git từ mã nguồn [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Remove public key from SSH's known_hosts file

```
ssh-keygen -R hostname
```

## Change the Shell in Mac OS X Terminal

```
chsh -s /bin/bash
```

Reference: [http://osxdaily.com/2012/03/21/change-shell-mac-os-x/](http://osxdaily.com/2012/03/21/change-shell-mac-os-x/)

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

## Create a new user account on \*NIX 

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
cat file | tr '\t' ' '
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



