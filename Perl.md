# Perl Notes
Ghi chép cá nhân khi học và sử dụng ngôn ngữ lập trình Perl.

## Array slicing

Reference: [http://perldoc.perl.org/perldata.html#Slices](http://perldoc.perl.org/perldata.html#Slices).

## Scan tất cả các vị trí trong xâu ký tự match với một regular expression
Sử dụng /g modifier và vòng lặp while để duyệt.

## Khai báo dữ liệu ngay trong perl script
Sử dụng từ khoá đặc biệt ```__DATA__```. Tham khảo: [http://perldoc.perl.org/perldata.html#Special-Literals](http://perldoc.perl.org/perldata.html#Special-Literals).
```perl
__DATA__
MLTSHQKKF*HDWFLSFKD*SNNYNSKQNHSIKDIFNRFNHYIYNDLGIRTIA
MLTSHQKKFSNNYNSKQNHSIKDIFNRFNHYIYNDLGIRTIA
MLTSHQKKFSNNYNSK*HDWFLSFKD*QNHSIKDIFNRFNHYIYNDLGIRTIA
```

## Regular expression - xác định vị trí cuối cùng của phần matching với pattern

Ví dụ xâu ký tự `str='0123456789'`. Ta cần match với `234`, khi đó vị trí cuối cùng được xác định là vị trí số 4. Sử dụng hàm pos trong perl. 

Tham khảo [http://stackoverflow.com/questions/832878/how-do-i-find-the-index-location-of-a-substring-matched-with-a-regex-in-perl](http://stackoverflow.com/questions/832878/how-do-i-find-the-index-location-of-a-substring-matched-with-a-regex-in-perl).


## Regular expression - match xâu ký tự với một pattern từ một vị trí trong câu

Vấn đề: match một xâu ký tự sau từ vị trí bất ký trong xâu. Cách đơn giản là lấy sub-string của xâu ký tự đó.
```perl
str = '0123456789';
```

## Convert ký tự hoa thành ký tự thường (và ngược lại)
Sử dụng ```lc expr``` và ```uc expr```.






