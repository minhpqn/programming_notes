# Shell script notes

Author: Pham Quang Nhat Minh

## If then else with string comparision

```
if [ "$Sourcesystem" = "XYZ" ]; then 
    echo "Sourcesystem Matched" 
else
    echo "Sourcesystem is NOT Matched $Sourcesystem"  
fi;
```

## For i from 1 to N

```for i in $(seq 1 $END); do echo $i; done```

## Set the current directory

```
cur_dir=`cd $(dirname $0) && pwd`
```

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

