#!/usr/bin/env python3
"""手順2（先行分）: 視野角3サブセットの75人版を GaitDatasetB-silh-train-png からコピー方式で構築する。
ryu 側は一切変更しない（読み取りのみ）。
assert ①〜⑦:
 ① 出力の被験者数 = 75（= train 全被験者）
 ※ source の空 view ディレクトリは作らない（ryu の F-delete.sh 末尾 `find -type d -empty -delete` と同じ扱い。③で ryu 側と集合一致を確認）
 ② 各被験者・各シーケンスに含まれる view が指定2視野角と厳密一致（source に存在するもののみ）
 ③ ryu の71人版に存在する被験者は、ファイル集合が同一
 ④ ③のファイルはすべてバイト一致（cmp）
 ⑤ 追加された被験者 = {018,036,054,072} ちょうど
 ⑥ 出力に指定外の view / 空ディレクトリが無い
 ⑦ manifest（件数・sha256）を書き出し、再実行時は既存出力と manifest が一致
"""
import os, sys, json, hashlib, shutil, filecmp
SRC = "/home/ryu/OpenGait/CASIA-B-png/GaitDatasetB-silh-train-png"
RYU = "/home/ryu/OpenGait/CASIA-B-png"
OUT = "/home/kera/incdino_data/CASIA-B-png-75"
SUBSETS = {"000-090-png": ("000", "090"), "000-180-png": ("000", "180"), "090-180-png": ("090", "180")}
EXPECT_ADDED = {"018", "036", "054", "072"}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def walk_files(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = p
    return out

subjects = sorted(os.listdir(SRC))
assert len(subjects) == 75, subjects
report = {"src": SRC, "out": OUT, "subsets": {}}
for name, views in SUBSETS.items():
    dst = os.path.join(OUT, name)
    ryu_dir = os.path.join(RYU, name)
    print(f"=== {name} views={views}")
    if os.path.exists(dst):
        print("  output exists -> verify only")
    else:
        for s in subjects:
            for seq in sorted(os.listdir(os.path.join(SRC, s))):
                for v in views:
                    sp = os.path.join(SRC, s, seq, v)
                    if os.path.isdir(sp) and os.listdir(sp):   # 空 view は作らない（ryu の F-delete.sh 末尾の空dir削除と同じ）
                        shutil.copytree(sp, os.path.join(dst, s, seq, v))
    # ①
    out_subj = sorted(os.listdir(dst)); assert out_subj == subjects, "①"
    # ② ⑥
    nfiles = 0
    for s in out_subj:
        for seq in os.listdir(os.path.join(dst, s)):
            vs = sorted(os.listdir(os.path.join(dst, s, seq)))
            src_vs = sorted(v for v in views if os.path.isdir(os.path.join(SRC, s, seq, v)) and os.listdir(os.path.join(SRC, s, seq, v)))
            assert vs == src_vs and set(vs) <= set(views), ("②⑥", s, seq, vs)
            for v in vs:
                fs = os.listdir(os.path.join(dst, s, seq, v)); assert fs, ("⑥ empty", s, seq, v)
                nfiles += len(fs)
    # ③ ④ ⑤
    ryu_subj = sorted(os.listdir(ryu_dir))
    assert set(out_subj) - set(ryu_subj) == EXPECT_ADDED, ("⑤", set(out_subj) - set(ryu_subj))
    ncmp = 0
    for s in ryu_subj:
        a = walk_files(os.path.join(ryu_dir, s)); b = walk_files(os.path.join(dst, s))
        assert a.keys() == b.keys(), ("③", s, len(a), len(b))
        for k in a:
            assert filecmp.cmp(a[k], b[k], shallow=False), ("④", s, k)
            ncmp += 1
    # ⑦
    allf = walk_files(dst)
    h = hashlib.sha256()
    for k in sorted(allf):
        h.update(k.encode()); h.update(sha256(allf[k]).encode())
    man = {"subjects": 75, "ryu_subjects": len(ryu_subj), "added": sorted(EXPECT_ADDED), "views": views,
           "n_files": nfiles, "n_files_cmp_with_ryu": ncmp, "filelist_sha256": h.hexdigest()}
    mp = os.path.join(OUT, f"manifest_{name}.json")
    if os.path.exists(mp):
        old = json.load(open(mp)); assert old == man, ("⑦ manifest mismatch", old, man)
    else:
        json.dump(man, open(mp, "w"), indent=2)
    report["subsets"][name] = man
    print("  OK", json.dumps(man))
json.dump(report, open(os.path.join(OUT, "step2_report.json"), "w"), indent=2, ensure_ascii=False)
print("STEP2_DONE")
