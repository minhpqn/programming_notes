"""Sentence segmentation for Japanese
"""
import nltk


jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')


def sent_tokenize(text):
    sentences = jp_sent_tokenizer.tokenize(text)
    if len(sentences) == 0:
        sentences.append(text)
    return sentences


if __name__ == "__main__":
    sentences = sent_tokenize("出身で片男波部屋に所属していた元大相撲力士。本名は福重二郎（ふくしげじろう）。身長175cm、体重155kg。最高位は東幕下8枚目。日本大学相撲部出身。")
    print(sentences)
