from pipeline.paths import Paths
import pandas as pd

class Corpus:

    def __init__(self, source_name):
        # get meta data
        self.paths = Paths()
        self.meta_data = pd.read_csv(self.paths.corpus_meta_data)




if __name__ == "__main__":
    corp = Corpus('all_the_news')
    idxs = [2, 67, 90, 62345]
    print(corp.meta_data.loc[idxs])
