import argparse
import pandas as pd
import nltk
import evaluate
import tqdm
import prompts


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str,
                    help="dataset to preprocess",
                    choices=['train','dev','test'],
                    default='train')
parser.add_argument("-llm_reranker", type=bool,
                    help="True if using LLM-based re-ranker",
                    default=True)
parser.add_argument("-threshold", type=restricted_float,
                    help="threshold value for selecting pos and neg examples for reranker fin-tuning",
                    default=0.35)

args = parser.parse_args()


# load data and select EN data
dataset = args.dataset
print(f" >> loading {dataset} data...")
df_train = pd.read_json(f"data/{dataset}.json").T
df_train_en = df_train[df_train.LANG == 'EN']

# extract CN and related KN
KN = []
for record in df_train_en.KN.to_list():
    KN.append([item.strip() for item in record.split('<EOS>') if item])

KN_CN = df_train_en.KN_CN.to_list()

# compute ROUGE scores between CN(predictions) over each KN(reference)
print(f"\n >> Computing ROUGE scores of {dataset} dataset...\n")
rouge = evaluate.load('rouge')
predictions = KN_CN
references = KN
results = []

for cn,kn in tqdm.tqdm(zip(predictions,references), total=len(predictions)):
    cn_sents = nltk.sent_tokenize(cn)
    # ROUGE between the entire cn and each KN sent
    rougel = rouge.compute(predictions=[cn]*len(kn), references=kn, use_aggregator=False)['rougeL']
    if len(cn_sents) > 1:   # if CN contains more than one sent
        rl_scores = []
        for sent in cn_sents:
            sent_rl = rouge.compute(predictions=[sent]*len(kn), references=kn, use_aggregator=False)['rougeL']
            rl_scores.append(sent_rl)
        rl_scores.append(rougel)    # evidence used in between two sentences
        results.append([max(values) for values in zip(*rl_scores)])   # taking max RL score for each sent
    else:
        results.append(rougel)  # CN with only one sent

# merging results with training dataset
train_rouge_df = {
    'PAIR_ID' : df_train_en.PAIR_ID.to_list(),
    'KN_CN' : KN_CN,
    'KN' : KN,
    'ROUGE' : results
}

df_results = pd.DataFrame(train_rouge_df)
df_train_scores = pd.merge(df_train, df_results[['PAIR_ID', 'ROUGE']], how='right', on=['PAIR_ID'])
df_train_scores.T.to_json(f"data/{dataset}_rl_scores.json", indent=4)   # saving scores


# ===============================================
# ========== RE-RANNKERS TRAINING DATA ==========
# ===============================================

# creating training set for reranker ft using the following format
# {"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str}
# For normal reranker fine-tuning, "prompt" entry is not necessary

print("\n >> Creating re-ranker training data...\n")

llm_reranker = args.llm_reranker 
THRESHOLD = args.threshold
prompt = prompts.LLM_RERANKER_FT_PROMPT

print(" >>PARAMS:")
print(f"-LLM Reranker: {llm_reranker}")
print(f"-THRESHOLD: {THRESHOLD}")
if llm_reranker:
    print(f"-PROMPT: {prompt}\n")

reranker_train_set = []
pair_ids = df_train_scores.PAIR_ID.to_list()
queries = df_train_scores.HS.to_list()  # HS as queries

kn_mul = []
for record in df_train_scores.KN.to_list():
    kn_mul.append([item.strip() for item in record.split('<EOS>') if item])
rouge_scores = df_train_scores.ROUGE.to_list()


for id, query, kn_list, scores in tqdm.tqdm(zip(pair_ids, queries, kn_mul, rouge_scores), total=len(queries)):
    pos = []
    neg = []
    pos_scores = []
    neg_scores = []
    # filtering results in pos and neg according to ROUGE scores
    for kn,score in zip(kn_list,scores):
        if score >= THRESHOLD:
            pos.append(kn)
            pos_scores.append(round(score*100, 3))
        else:
            neg.append(kn)
            neg_scores.append(round(score*100, 3))

    if len(neg) == 0:   # randomly assign a negative example if none is present
        query_lang = df_train_scores[df_train_scores.HS == query].LANG.item() # get query language
        # pick all the entries in 'query_lang' with the exception of the query we are working on
        examples = df_train_scores[~(df_train_scores.HS == query) & (df_train_scores.LANG == query_lang)]
        # randomply pick a KN as a negative example
        random_neg_example = examples.sample(n=1, random_state=42)
        random_kn = random_neg_example['KN'].item().split('<EOS>')[0]
        neg.append(random_kn)
        random_score_kn = random_neg_example['ROUGE'].item()[0]
        neg_scores.append(random_score_kn)

    if llm_reranker:
        reranker_train_set.append({"query" : query, "pos" : pos, "neg" : neg, "pos_scores" : pos_scores, "neg_scores" : neg_scores, "prompt" : prompt})
    else:
        reranker_train_set.append({"query" : query, "pos" : pos, "neg" : neg, "pos_scores" : pos_scores, "neg_scores" : neg_scores})

pd.DataFrame(reranker_train_set).to_json(f"data/reranker_{dataset}{'_LLM' if llm_reranker else ''}.jsonl", orient='records', lines=True, force_ascii=False)