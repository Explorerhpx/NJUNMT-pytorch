import os
import subprocess
from subprocess import DEVNULL
from tqdm import tqdm

from sacrebleu.metrics import BLEU

DETRUECASE_PL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/metric/scripts/recaser/detruecase.perl")
DETOKENIZE_PL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/metric/scripts/tokenizer/detokenizer.perl")
ZH_TOKENIZER_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/metric/scripts/tokenizer/tokenizeChinese.py")

def postprocess_cmd(stdin,tgt_lang):
    cmd_detrucase = subprocess.Popen(["perl", DETRUECASE_PL], stdin=stdin, stdout=subprocess.PIPE)
    cmd_postprocess = subprocess.Popen(["perl", DETOKENIZE_PL, "-q", "-l", tgt_lang],
                                           stdin=cmd_detrucase.stdout, stdout=subprocess.PIPE)
    return cmd_postprocess

if __name__ == "__main__":
    print("#### Current vision is for English only ####")

    # post process
    print("Post Process")
    with open("/data1/huangpx/FDMT/experiment-data/cwmt17+bt+ft+seed_page/AV_seed_page.btft.en","r") as tf:
        trans_post = postprocess_cmd(stdin=tf,tgt_lang="en")
    trans_post = trans_post.communicate()[0].decode("utf-8").split("\n")

    # sentence sacre_bleu
    print("Calculate BLEU")
    with open("/home/data_ti6_c/huangpx/data/AV/AV_seed_page_sentenced.en","r") as rf:
        refs = list(rf.readlines())
    refs = [[i] for i in refs]

    # en
    bleu = BLEU(lowercase=True,trg_lang = "en",effective_order=True)
    # # zh
    # # not post process
    # bleu = BLEU(tokenize="none",trg_lang = "zh",effective_order=True)
    sent_scores = []
    for hyp,ref in tqdm(zip(trans_post,refs),total=len(trans_post)):
        sent_scores.append(bleu.sentence_score(hyp,ref).score)
    with open("/data1/huangpx/FDMT/experiment-data/cwmt17+bt+ftr+seed_page/AV_seed_page.btftt.en.sent_bleu","w") as wf:
        wf.write(str(sent_scores))
