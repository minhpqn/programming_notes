# Mẹo sử dụng LaTeX

## Cite không cần ngoặc

Dùng lệnh `\newcite`

Ví dụ:

```
~\newcite{li2014personal,hirano2015user}
```

## Cách submit file pdf sinh ra bằng LaTeX lên arxiv

Thêm vào dòng sau đây:

```
\pdfoutput=1
\documentclass{article}
\usepackage[final]{pdfpages}
\begin{document}
\includepdf[pages=1-last]{paper.pdf}
\end{document}
```

Lưu lại file trên dưới tên `main.tex` sau đó submit hai files `main.tex` và `paper.pdf` lên arxiv.

Tham khảo: [How to Bypass arXiv LaTeX-generated PDF Detection in Six Lines](https://mssun.me/blog/how-to-bypass-arxiv-latex-generated-pdf-detection-in-six-lines.html)

