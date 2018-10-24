"""
Test case-frame analysis with pyknp
"""
from pyknp import KNP

jumanpp_opt = True
knp_option = "-case -bnsttab -normal -anaphora"
# knp_option = "knp -case -tab -normal"  # for case-frame analysis
knp = KNP(jumanpp=jumanpp_opt, option=knp_option)
knp = KNP(jumanpp=True)
# res = knp.parse("私も行きたいですよ。")
# res = knp.parse("私もサンフランシスコに行きたいんです。")
# res = knp.parse("彼女はきれいな人です。")
# res = knp.parse("夏休みにはどこに行きたいですか。")
res = knp.parse("太郎が食べたバナナは古かった。")
# res = knp.parse("いつ就任しましたか。")
# res = knp.parse("私は学生です。")
res = knp.parse("私も！")
# res = knp.parse("私は学生ですね。")
res = knp.parse("私は１８歳で学生ですね。")

for bnst in res.bnst_list(): 
    print(bnst.midasi)
    print(bnst.fstring)
    print("Tags:...")
    for t in bnst._tag_list:
        print("tid: {}, fstring: {}".format(t.tag_id, t.fstring))
        if t.pas is not None:
            print("-- Pas")
            print(t.pas)
            print("tag id:", t.pas.tid)
            print("tag pred_name: {}, normalized_repname: {}, head_repname: {}".format(t.pred_repname, t.normalized_repname, t.head_repname))
            for case in t.pas.arguments:
                print("  case: {}".format(case))
                for arg in t.pas.arguments[case]:
                    print("   sid: {}, tid: {}, eid: {}, midasi: {}, flag: {}, sdist: {}".format(arg.sid, arg.tid, arg.eid, arg.midasi, arg.flag, arg.sdist))

    print()






