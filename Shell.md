# Shell script notes

## Create tar.gz file

```
tar -cvzf file.tar.gz file_list
```

## Lấy thư mục hiện tại

```
cur_dir=`cd $(dirname $0) && pwd`
```

## Vòng lặp for trong shell script

```
for var in 0 1 2 3 4 5 6 7 8 9
do
   echo $var
done
```

## Kiểm tra số lượng đầu vào của một shell script

```shell
if [ $# -lt 1 ]
then
   echo "You must give at least one argument"
   exit 1
fi
````

