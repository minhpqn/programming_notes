#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("./")
import unittest
import re
import six
import unicodedata
from sub_process import SubProcess


class Chunker():
    def __init__(self):
        self.juman_process = SubProcess("juman")
        self.knp_process = SubProcess("knp -tab -dpnd-fast")
        self.pattern = r'EOS'

    def get_juman_line(self, sentence):
        assert (isinstance(sentence, six.text_type)), "input sentence is not unicode type"
        assert (self.__check_full_width(sentence)), "input contain half-width character"
        return self.juman_process.query(sentence, pattern=self.pattern)

    def get_knp_line(self, sentence):
        assert (isinstance(sentence, six.text_type)), "input sentence is not unicode type"
        assert (self.__check_full_width(sentence)), "input contain half-width character"
        juman_lines = self.juman_process.query(sentence, pattern=self.pattern)
        juman_str = "%s%s" % (juman_lines, self.pattern)
        return self.knp_process.query(juman_str, pattern=self.pattern)

    def get_noun_phrase_normalize(self, sentence):
        result = []
        knp_line = self.get_knp_line(sentence)
        for line in knp_line.split("\n"):
            if line.strip() and line.startswith(u"*") and u"<体言>" in line:
                match = re.search(u"<正規化代表表記:(.*?)>", line)
                if match:
                    result.append(re.sub(u"/[^\+]*.", u"", match.group(1)))
        return result

    def __check_full_width(self, sentence):
        for character in sentence:
            if unicodedata.east_asian_width(character) == "Na" or unicodedata.east_asian_width(character) == "H":
                print( "Half width character: {} in sentence {}".format(character.encode('utf-8'),
                                                                       sentence.encode('utf-8')) )
                return False
        return True

    def __get_noun_pharse_extend(self, knp_str):

        # knp_lines = self.get_knp_line(sentence)
        # knp_str = "%s%s" % (knp_lines, r'EOS')

        result = []
        tmp_str = ""

        (np_flag, adnominal_flag, mem) = (0, 0, "")
        for line in knp_str.split("\n"):
            ls = line.split(" ")

            # new bunsetsu
            if re.match(u"[\*E]", line):
                # adnominal_flag && "体言" => print mem
                if adnominal_flag and re.match(u"\*.*<体言>", line) and not re.search(u"<括弧始>", line):
                    # print u"{0}".format(mem)
                    tmp_str += mem
                # flag off
                elif np_flag:
                    # print("\n")
                    result.append(tmp_str)
                    tmp_str = ""
                    np_flag = 0
                adnominal_flag = 0

            # new bunsetsu is "体言" => np_flag on
            if re.match(u"\*.*<体言>", line):
                np_flag = 1
                # new bunsetsu is "ノ格" => adnominal_flag on
                if re.search(u"<係:ノ格>", line) and not re.search(u"<括弧終>", line):
                    (adnominal_flag, mem) = (1, "")

            # read a morpheme line if flag on
            if (np_flag or adnominal_flag) and len(ls) > 4:
                if re.search(u"内容語>|<複合←>", line) or re.match(u"接[頭尾]辞", ls[3]):
                    # print(u"{0}".format(ls[0]))
                    tmp_str += ls[0]
                elif adnominal_flag:
                    if re.search(u"<文節始>", line):
                        adnominal_flag = 0
                    else:
                        mem += ls[0]
        return result

    def __get_noun_phrase_simple(self, knp_str):
        # juman_lines = self.juman_process.query(sentence, pattern=r'EOS')
        # juman_str = "%s%s" % (juman_lines, r'EOS')
        # knp_lines = self.knp_process.query(juman_str, pattern=r'EOS')
        # knp_str = "%s%s" % (knp_lines, r'EOS')

        surface = ""
        lst_np = []
        has_np = False
        for line in knp_str.split("\n"):
            if line.startswith(u"*") and surface:
                lst_np.append(surface)
                surface = ""
                has_np = False

            if re.search(u"^\*.*<体言>", line):
                has_np = True

            if has_np and re.search(u"内容語>|<複合←>", line):
                surface = "%s%s" % (surface, line.split(" ")[0])
        if surface:
            lst_np.append(surface)

        return lst_np

    def get_noun_phrase(self, sentence, extend=False):

        knp_lines = self.get_knp_line(sentence)
        knp_str = "%s%s" % (knp_lines, r'EOS')

        result = []
        result.extend(self.__get_noun_phrase_simple(knp_str))

        if extend:
            result.extend(self.__get_noun_pharse_extend(knp_str))
            result = list(set(result))

        return result

    def get_noun_phrase_with_position(self, sentence):

        knp_lines = self.get_knp_line(sentence)
        knp_str = "%s%s" % (knp_lines, r'EOS')

        result = []
        lst_np = self.__get_noun_phrase_simple(knp_str)

        last_idx = 0
        for np in lst_np:
            np_idx = sentence.find(np, last_idx)
            if np_idx == -1:
                print( "Error noun pharse: {} can not be found in sentence {}".format(np.encode('utf-8'),
                                                                                     sentence.encode('utf-8')) )
            else:
                last_idx = np_idx + len(np)
                result.append(dict(surface=np, start=np_idx, end=last_idx - 1, length=len(np)))

        return result

    def get_all_noun_phrase(self, sentence):

        knp_lines = self.get_knp_line(sentence)
        knp_str = "%s%s" % (knp_lines, r'EOS')

        result = []
        lst_np = self.__get_noun_phrase_simple(knp_str)
        result.extend(lst_np)

        simple_np_with_position = []
        last_idx = 0
        for np in lst_np:
            np_idx = sentence.find(np, last_idx)
            if np_idx == -1:
                print("Error noun pharse: {} can not be found in sentence {}".format(np.encode('utf-8'),
                                                                                     sentence.encode('utf-8')))
            else:
                last_idx = np_idx + len(np)
                simple_np_with_position.append(dict(surface=np, start=np_idx, end=last_idx - 1, length=len(np)))

        result.extend(self.__get_noun_pharse_extend(knp_str))
        result = list(set(result))

        return result, simple_np_with_position


class ChunkerTest(unittest.TestCase):
    def test(self):
        chunker = Chunker()

        self.assertSetEqual(set( chunker.get_noun_phrase(
            u"ドミニカ国の首都ロゾーにあるモルヌ・トロワ・ピトン国立公園は、トロワ・ピトン山を中心とした火山地帯です。", extend=True) ),
            {u"ドミニカ国", u"首都ロゾー", u"ドミニカ国の首都ロゾー", u"モルヌ・トロワ・ピトン国立公園", u"トロワ・ピトン山", u"中心", u"火山地帯"})

        self.assertSetEqual(set( chunker.get_noun_phrase(
            u"ドミニカ国の首都ロゾーにあるモルヌ・トロワ・ピトン国立公園は、トロワ・ピトン山を中心とした火山地帯です。") ),
            {u"ドミニカ国", u"首都ロゾー", u"モルヌ・トロワ・ピトン国立公園", u"トロワ・ピトン山", u"中心", u"火山地帯"})

        self.assertSetEqual(set( chunker.get_noun_phrase(
            u"ぼくたちと駐在さんの７００日戦争、ＦＣ２の頃から好きだったんだ。", extend=True) ),
            {u"ぼく", u"駐在", u"７００日", u"戦争", u"ＦＣ", u"頃", u"ぼくたち", u"駐在さんの７００日", u"ＦＣ２の頃"})

        self.assertSetEqual(set( chunker.get_noun_phrase(
            u"ぼくたちと駐在さんの７００日戦争、ＦＣ２の頃から好きだったんだ。") ),
            {u"ぼく", u"駐在", u"７００日", u"戦争", u"ＦＣ", u"頃"})

        print(chunker.get_noun_phrase("東京の日暮里で生まれ高校卒業するまで同じところに住んでいました。", extend=True))


if __name__ == '__main__':
    unittest.main()
