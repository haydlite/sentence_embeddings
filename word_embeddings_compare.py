import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import sys

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)


def find_closest_pair(embed1, embed2, snippets1, snippets2):
  best_score = float("-inf")
  pair = []
  for i, em1 in enumerate(embed1):
    for i2, em2 in enumerate(embed2):
      if(i == i2 or snippets1[i] == snippets2[i2]): continue
      sim_score = np.inner(em1,em2)
      if(sim_score > best_score):
        pair = [i, i2]
        best_score = sim_score
  return [best_score, snippets1[pair[0]], snippets2[pair[1]]]

def find_pair_close_to_target(target, embed1, embed2, snippets1, snippets2):
  closest_score = float("inf")
  pair = []
  for i, em1 in enumerate(embed1):
    for i2, em2 in enumerate(embed2):
      if(i == i2): continue
      sim_score = np.inner(em1,em2)
      if(abs(target-sim_score) < abs(target-closest_score)):
        pair = [i, i2]
        closest_score = sim_score
  return [closest_score, snippets1[pair[0]], snippets2[pair[1]]]

# 'same' boolean indicates that we are comparing within same corpus, not between corpora
def compute_sim_score(embeds1, embeds2, same=False):
    sim_list = []
    for i, em1 in enumerate(embeds1):
        for i2, em2 in enumerate(embeds2):
            if (i == i2 and same): continue
            sim_list.append(np.inner(em1, em2))
    cos_sim_avg = sum(sim_list) / len(sim_list)
    return cos_sim_avg

# Create list of "snippets"
def lst_to_snippets(lst):
  snippets = []
  for article in lst:
    s = ''
    for bit in article.split():
      if bit != '...':
        s += bit + ' '
      else:
        snippets.append(s)
        s = ''
    snippets.append(s)
  return snippets


if __name__ == "__main__":
    set_of_articles1 = sys.argv[1]
    set_of_articles2 = sys.argv[2]

    #1. Read file as panda dataframe
    db1 = pd.read_csv(set_of_articles1)
    db2 = pd.read_csv(set_of_articles2)

    # Create list with each article = an element
    lst1 = db1.iloc[:, 1].tolist()
    lst2 = db2.iloc[:, 1].tolist()

    snippets1 = lst_to_snippets(lst1)
    snippets2 = lst_to_snippets(lst2)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeds1 = session.run(embed(snippets1))
        embeds2 = session.run(embed(snippets2))

    print('Comparing snippets within ' + set_of_articles1 + ' with themselves...')
    sim1_1 = compute_sim_score(embeds1, embeds1, same=True)
    print('Within ' + set_of_articles1 + 'snippets have an average similarity score of '
          + str(sim1_1))

    print('Comparing snippets within ' + set_of_articles1 + ' with themselves...')
    sim2_2 = compute_sim_score(embeds1, embeds2, same=True)
    print('Within ' + set_of_articles1 + 'snippets have an average similarity score of '
          + str(sim2_2))

    print('Comparing snippets between ' + set_of_articles1 + ' and ' + set_of_articles2 + '...')
    sim1_2 = compute_sim_score(embeds1, embeds2, same=False)
    print(set_of_articles1 + "'s and " + set_of_articles2 + "'s snippets have an average similarity score of "
          + str(sim1_2))

    print('Finding pairs most representative of the above similarity scores...')
    print(find_pair_close_to_target(sim1_1, embeds1, embeds1, snippets1, snippets1))
    print(find_pair_close_to_target(sim2_2, embeds2, embeds2, snippets2, snippets2))
    print(find_pair_close_to_target(sim1_2, embeds1, embeds2, snippets1, snippets2))

    print('Finding most similar pairs in-set and between-sets...')
    print(find_closest_pair(embeds1, embeds1, snippets1, snippets1))
    print(find_closest_pair(embeds2, embeds2, snippets2, snippets2))
    print(find_closest_pair(embeds1, embeds2, snippets1, snippets2))