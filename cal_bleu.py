import argparse

from src.metric.bleu_scorer import SacreBLEUScorer

parser = argparse.ArgumentParser()
parser.add_argument('-ref','--reference_path', type=str,
                    help="reference files path")
parser.add_argument('-tr','--translation_file', type=str,
                    help="translation file path")
parser.add_argument('-n','--num_refs', type=int, default=1,
                    help="number of reference files")
parser.add_argument('--lang_pair','-l',type=str, default="zh-en",
                    help="language pair")
parser.add_argument('--sacrebleu_args',type=str, default="",
                    help="sacrebleu arguments, e.g. \"[\'-tok none\']\", \"[\'-lc\']\"")
parser.add_argument("--postprocess","-post", dest='postprocess', action='store_true',
                    help="whether to detrucase and detokenize")

parser.set_defaults(postprocess=False)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    sacrebleu_args = eval(args.sacrebleu_args)[0] if args.sacrebleu_args else None
    bleu_scorer = SacreBLEUScorer(reference_path=args.reference_path,
                                    num_refs=args.num_refs,
                                    lang_pair=args.lang_pair,
                                    sacrebleu_args=sacrebleu_args,
                                    postprocess=args.postprocess
                                    )
    
    with open(args.translation_file,"r") as f:
        print(bleu_scorer.corpus_bleu(f))
        