# Ghi ghép khi học/làm việc với ngôn ngữ Java

## Tạo file jar thực thi được từ dòng lệnh

```
jar cfe evaluation.jar org.vlsp.ner.VLSP2018MUCEvaluation -C bin/ .
```

## Biên dịch project từ dòng lệnh

```
javac -d bin -sourcepath src/ src/com/example/test/Main.java
```

Reference: [https://kipalog.com/posts/Bien-dich-java-project-voi-command-line](https://kipalog.com/posts/Bien-dich-java-project-voi-command-line)

## Read a text file  line by line

```
try(BufferedReader is = new BufferedReader(new FileReader(fileName))) {
    String line;
    while ( (line = is.readLine()) != null ) {
	      // Statements
   }
} catch(IOException e) {
   e.printStackTrace();
}
```

## Add an element in to a list in Java

Use ```add``` method.

## Iterate a list in Java

Tham khảo bài viết: [Five (5) ways to Iterate Through Loop in Java](http://crunchify.com/how-to-iterate-through-java-list-4-way-to-iterate-through-loop/)

## Indicate Java classpath with javac

Use ```-cp```

Ví dụ:

```
java -cp ".:/Users/minhpham/nlp/lib/stanford-corenlp-full-2015-12-09/*" CorenlpDemo
```

## Import a class in library to Java

```
import java.io.File;
import java.io.IOExeption;
```

## Initialize an array of strings

```
String[] names = {"Ankit","Bohra","Xyz"};
```

## For loop in Java

```
for(initialization; Boolean_expression; update) {
	//Statements
}

for(int j = 1; j <= 1024; j = j * 2) {
	System.out.println(j);
}
```

## Length of an array in Java

```
String[] string_array = {"A", "B", "C"};
System.out.println("Java array length is %d", string_array.length);
```



