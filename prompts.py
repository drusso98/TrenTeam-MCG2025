LLM_RERANKER_FT_PROMPT = """Given an hateful content A and a possible argument B against it, \
determine whether the argument is an effective reply \
providing a prediction of either 'Yes' or 'No'.""" 

RERANK_BASED_GEN_PROMPT = """You will be provided with a hateful comment (hate speech) and two sentences comprising arguments against the comment (knowledge). \
Generete a reply to the hateful content using only the information present in the knowledge. Reply in the following language: {language} 

Hate speech: 
{hate_speech} 

Knownledge:
{knowledge} 

Reply:"""

E2E_GEN_PROMPT = """You will be provided with a hateful comment (hate speech) and {nof_sent} sentences comprising arguments against the comment (knowledge). \
Select the most effective sentences and use them to generete a reply to the hateful content. Reply in the following language: {language} 

Hate speech: 
{hate_speech} 

Knownledge:
{knowledge} 

Reply:"""