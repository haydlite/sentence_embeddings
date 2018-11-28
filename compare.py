import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import sys
import os

import shelve

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)

# create all csv in curr directory + sub directories
def getListOfCsv(dirName):

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfCsv(fullPath)
        if fullPath.endswith('csv'):
            allFiles.append(fullPath)

    return allFiles

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

if __name__ == '__main__':
    # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)

    corpora_as_df = {}
    corpora_as_snippets = {}
    corpora_embeddings = {}

    listOfFiles = getListOfCsv(os.getcwd())
    # we exclude the globe and observer.csv that we worked off of as samples
    listOfFiles = listOfFiles[2:]

    for file in listOfFiles:
        last_slash = file.rfind('/')
        second_to_last_slash = file[:last_slash].rfind('/')

        name = file[last_slash + 1: -4]
        newspaper = file[second_to_last_slash + 1: last_slash]

        if newspaper not in corpora_as_df:
            corpora_as_df[newspaper] = {}

        # TODO: Stop reading in data with pandas when unneccesary.
        corpora_as_df[newspaper][name] = pd.read_csv(file)

    for newspaper in corpora_as_df:
        if newspaper not in corpora_as_snippets:
            corpora_as_snippets[newspaper] = {}

        nw = corpora_as_df[newspaper]
        for df in nw:
            lst = nw[df].iloc[:, 1].tolist()
            snippets = lst_to_snippets(lst)

            corpora_as_snippets[newspaper][df] = snippets

    # TODO: Randomly sample snippets from corpora too large to wholly embed.

    # for newspaper in corpora_as_snippets:
    #     print("Embedding " + newspaper + " corpora...")
    #     if newspaper not in corpora_embeddings:
    #         corpora_embeddings[newspaper] = {}
    #     nw = corpora_as_snippets[newspaper]
    #     for df in nw:
    #         if df not in corpora_embeddings[newspaper]:
    #             print("Embedding " + df + "...")
    #             with tf.Session() as session:
    #                 # session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #                 embedding = session.run(embed(nw[df]))
    #             corpora_embeddings[newspaper][df] = embedding
    #
    # s = shelve.open('shelf')
    #
    # try:
    #     s['corpora_embeddings'] = corpora_embeddings
    # finally:
    #     s.close()