# Some notes about using Linux/Unix

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



