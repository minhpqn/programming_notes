# Some notes about using Linux/Unix

Tác giả: Phạm Quang Nhật Minh

## Cách copy để đồng bộ 2 thư mục

Có 2 thư mục A, B trong đó có các sub-folder trong B trùng tên với A. Tôi merge nội dung từ B vào A và không ghi đè các thư mục trùng tên.

Sử dụng lệnh `rsync`

```
rsync -av --ignore-existing B/ A/
```

Chú ý:
- Tùy chọn `--ignore-existing` để không ghi đè các file đã có trong A
- Các dấu `/` quan trọng vì nó sẽ chỉ định lệnh `rsync` merge nội dung B với A chứ không lưu dữ liệu B vào thư mục con trong A.

## Shutdown server

```
shutdown now
```

## Cài đặt để ssh bằng public key và không cần password

Edit file sshd_config

```
sudo emacs /etc/ssh/sshd_config
```

Tìm và set các giá trị sau trong file trên

```
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
```

Restart the SSH daemon

```
sudo systemctl restart sshd
```

## Cài đặt để thực thi lệnh sudo không cần nhập password

Sử dụng lệnh sau đây

```
sudo visudo
```

Thêm dòng sau vào cuối file

```
username ALL=(ALL) NOPASSWD:ALL
```

Trong đó thay username bằng tên user thực tế.


## Xác định các phân vùng chưa được mount

Sử dụng một trong các lệnh sau:

```
lsblk
```

Hoặc

```
sudo fdisk -l
```

## Thiết lập để mount ổ cứng tự động

Thêm dòng sau vào file `/etc/fstab`

```
/dev/nvme2n1  /mnt/disk1 ext4  defaults  0  2
```

Có thể thay tên thiết bị bằng UUID tương ứng

```
UUID=xxxx-xxxx  /mnt/disk1  ext4  defaults  0  2
```

Xác định UUID của thiết bị bằng một trong các lệnh sau

```
lsblk -f
```

hoặc

```
sudo blkid
```


## Permission đúng của file pem

```
chmod 400 path/to/pem/file
chmod 600 path/to/pem/file
```

## Xử lý khi mount phân vùng chưa được format

Thông báo lỗi

```
mount: wrong fs type, bad option, bad superblock on /dev/sdb,
       missing codepage or helper program, or other error
       In some cases useful info is found in syslog - try
       dmesg | tail  or so
```

Xử lý như sau:

```
mkfs -t ext4 /dev/sdb1
```

Tham khảo: https://ex1.m-yabe.com/archives/3457

## Xác định version của ubuntu

```
lsb_release -a
```

## List directory và sắp xếp theo size

```
du -h --max-depth=1 | sort -hr
```

Reference: [https://linuxconfig.org/list-all-directories-and-sort-by-size](https://linuxconfig.org/list-all-directories-and-sort-by-size)

## Cách sử dụng soffice trên MAC OS X

Tham khảo: [https://gist.github.com/minhpqn/20692891235cd5dfa46c0891682e31d9](https://gist.github.com/minhpqn/20692891235cd5dfa46c0891682e31d9)


## Giải nén file tar.bz2

```
tar -xf archive.tar.bz2
```

## Check encoding của file

```
file -i (trên Linux)
file -I (OS X)

```

Tham khảo: [https://stackoverflow.com/questions/805418/how-can-i-find-encoding-of-a-file-via-a-script-on-linux](https://stackoverflow.com/questions/805418/how-can-i-find-encoding-of-a-file-via-a-script-on-linux)

## Cách check domain đã đăng ký trên aws

Tham khảo: [https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/domain-view-status.html](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/domain-view-status.html)

## Nén file với password

```
zip -r -e  backend.zip  backend
```

## Cách extract file rar trên Linux

Cài đặt unrar

```
sudo apt-get install unrar
```

Giải nén với unrar

```
unrar x file.rar
```

Tham khảo: [https://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/](https://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/)

## Sort all directories based on their size

```
du -sh -- *  | sort -rh  # Files and directories, or
du -sh -- */ | sort -rh  # Directories only
```

Tham khảo [https://unix.stackexchange.com/questions/106330/sort-all-directories-based-on-their-size](https://unix.stackexchange.com/questions/106330/sort-all-directories-based-on-their-size)

## Trích ra x-megabytes đầu từ 1 file dữ liệu lớn

Dùng `head -c` hoặc lệnh `dd`

```
# Lấy 1000 MB từ file ja.txt
head -c 1000M ja.txt > ja1000M.txt
# Dùng lệnh dd
dd if=ja.txt count=1000 bs=1M > ja1000M.txt
```

Tham khảo: https://stackoverflow.com/questions/12221387/how-to-extract-the-first-x-megabyte-from-a-large-file-in-unix-linux

## Xóa bớt một kí tự space trong cột thứ 2 bằng dòng lệnh

Bài toán: có 1 file chia thành 2 cột cách nhau bằng dấu tab. Lập trình viên sơ ý nên đầu cột thứ 2 có thêm dấu space. Hãy xóa dấu space ở đầu của cột thứ 2 chỉ bằng lệnh trong Unix.

Đầu vào:

```
η	 i
ή	 ί
ο	 o
αμ	 a m
αν	 a n
```

Đầu ra: vẫn 2 cột trên nhưng dấu space đã được xóa

## Lấy content của file từ dòng thứ n

```
awk 'NR>4' file
tail -n +10 file
```

Tham khảo: [https://stackoverflow.com/questions/25678863/how-to-display-all-the-lines-from-the-nth-line-of-a-file-in-unix](https://stackoverflow.com/questions/25678863/how-to-display-all-the-lines-from-the-nth-line-of-a-file-in-unix)

## Getting back old copy paste behaviour in tmux, with mouse

```
prefix : set -g mouse off
```

Tham khảo: [https://stackoverflow.com/questions/17445100/getting-back-old-copy-paste-behaviour-in-tmux-with-mouse](https://stackoverflow.com/questions/17445100/getting-back-old-copy-paste-behaviour-in-tmux-with-mouse)

## Khắc phục lỗi Xid: 79, GPU has fallen off the bus

Lỗi này là do GPU quá nóng. Có thể tạm khắc phục bằng cách setup `pcie_aspm=off`.

Edit file `/etc/default/grub` và sửa dòng `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"`.

Tham khảo: [https://askubuntu.com/questions/68372/how-do-i-add-pcie-aspm-force-to-my-boot-options](https://askubuntu.com/questions/68372/how-do-i-add-pcie-aspm-force-to-my-boot-options)


## Cut cột đầu của một file để dán vào file khác

```
paste <(cut -f 2 o2.csv) g3temp.csv > tmp && mv tmp g3temp.csv
```

Tham khảo: [https://stackoverflow.com/questions/26753270/cut-column-from-some-file-1-paste-to-a-file-2-and-write-result-to-file-2](https://stackoverflow.com/questions/26753270/cut-column-from-some-file-1-paste-to-a-file-2-and-write-result-to-file-2)

## Change default shell to zsh

```
chsh -s $(which zsh)
```

Tham khảo: [https://askubuntu.com/questions/131823/how-to-make-zsh-the-default-shell](https://askubuntu.com/questions/131823/how-to-make-zsh-the-default-shell)

## Convert .mov to .mp4 with ffmpeg

```
ffmpeg -i my-video.mov -vcodec h264 -acodec mp2 my-video.mp4
```

Reference: [https://mrcoles.com/convert-mov-mp4-ffmpeg/](https://mrcoles.com/convert-mov-mp4-ffmpeg/)


## Show all lines except the first line of a text file

```
sed '1d' file.txt
```

Tham khảo: [https://unix.stackexchange.com/questions/55755/print-file-content-without-the-first-and-last-lines](https://unix.stackexchange.com/questions/55755/print-file-content-without-the-first-and-last-lines)

## Disable auto-title in zsh

Thêm dòng `DISABLE_AUTO_TITLE=true` vào file `~/.zshrc`.

## Set-up zsh shell

[Jazz Up Your “ZSH” Terminal In Seven Steps — A Visual Guide](https://www.freecodecamp.org/news/jazz-up-your-zsh-terminal-in-seven-steps-a-visual-guide-e81a8fd59a38/)

## Set character encoding for shell

```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
```

## Show the usage of GPU

Cách 1:

```
# Show the window every 1 second
nvidia-smi -l 1
```

Cách 2 (tốt hơn)

```
watch -d -n 1.0 nvidia-smi
```

Cập nhật cửa sổ sau 0.5 giây và không print liên tiếp trên terminal.

## Cách SSH tới remote server sử dụng ngrok

Xem hướng dẫn tại [https://medium.com/@byteshiva/ssh-into-remote-linux-by-using-ngrok-b8c49b8dc3ca](https://medium.com/@byteshiva/ssh-into-remote-linux-by-using-ngrok-b8c49b8dc3ca)

## List files and sort by sizes

```
ls --sort=size -lh
```

hoặc

```
ls -S -lh
```

## Add public key để đăng nhập tự động

Bổ sung SSH kay vào file `authorized_keys` trong thư mục `~/.ssh`

## Show combined file size in Mac OS X

Command + Option + I instead of Command + I

## NVIDIA NVML Driver/library version mismatch

Tham khảo: [https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)

## Clear command line history

```
history -c
```

Tham khảo: [https://www.ostechnix.com/how-to-clear-command-line-history-in-linux/](https://www.ostechnix.com/how-to-clear-command-line-history-in-linux/)

## How to check free disk space in Linux

```
df -ha
```

Tham khảo: [How to check free disk space in Linux](https://opensource.com/article/18/7/how-check-free-disk-space-linux)

## Check địa chỉ ip của AWS server

```
curl http://checkip.amazonaws.com
```

Tham khảo: [https://forums.aws.amazon.com/thread.jspa?threadID=216352](https://forums.aws.amazon.com/thread.jspa?threadID=216352)

## Rename a pane in tmux

Ctrl + b + ,

Reference: [https://stackoverflow.com/questions/40234553/how-to-rename-a-pane-in-tmux](https://stackoverflow.com/questions/40234553/how-to-rename-a-pane-in-tmux)

## Đặt alias cho các ssh connection

Sửa file `~/.ssh/config`

Content

```
Host xxx
    Hostname xxx
    User xxx
```

## Check Ubuntu version

```
lsb_release -a
```

## Check CUDA version

```
cat /usr/local/cuda/version.txt
nvcc --version
```

## Vào một session cụ thể trong tmux

```
tmux a -t <session_name>
```

## Liệt kê các session trong tmux

```
tmux ls
```

## Đổi tên session name trong tmux

Ctrl + B, $

## Tạo file tar.gz trong linux

```
tar -zcvf tar-archive-name.tar.gz source-folder-name
```

Reference: [https://www.zyxware.com/articles/2009/02/26/how-to-create-and-extract-a-tar-gz-archive-using-command-line](https://www.zyxware.com/articles/2009/02/26/how-to-create-and-extract-a-tar-gz-archive-using-command-line)

## Convert multiple spaces into one

Using tr

```
tr -s " " < file
```

or using awk

```
awk '{$2=$2};1' file
```

Reference: [https://unix.stackexchange.com/questions/145978/replace-multiple-spaces-with-one-using-tr-only/145979](https://unix.stackexchange.com/questions/145978/replace-multiple-spaces-with-one-using-tr-only/145979)


## Xác định đường dẫn của một process

```
pwdx <process_id>
```

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

Gõ `Enter, ~, .`

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



