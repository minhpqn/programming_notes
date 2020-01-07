import MeCab

tagger = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")


def tokenize(raw_sentence):
    # http://testpy.hatenablog.com/entry/2016/10/04/010000
    result = tagger.parse(raw_sentence)
    words = result.split()
    if len(words) == 0:
        return ""
    if words[-1] == "\n":
        words = words[:-1]
    return " ".join(words)


raw_sentence = "8月3日に放送された「中居正広の金曜日のスマイルたちへ」(TBS系)で、1日たった5分でぽっこりおなかを解消するというダイエット方法を紹介。キンタロー。にも密着。"

print(raw_sentence)
print(tokenize(raw_sentence))