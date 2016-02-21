# Shell script notes

## Kiểm tra số lượng đầu vào của một shell script

```shell
if [ $# -lt 1 ]
then
   echo "You must give at least one argument"
   exit 1
fi
````

