

class Sources:

    urls = {
        'Associated Press': 'https://apnews.com/',
        'Breitbart News': 'http://www.breitbart.com',
        'Buzzfeed': 'https://www.buzzfeed.com',
        'CNN': 'http://us.cnn.com',
        'Fox News': 'http://www.foxnews.com',
        'MSNBC': 'http://www.msnbc.com',
        'National Review': 'https://www.nationalreview.com/',
        'Politico': 'https://www.politico.com',
        'Reuters': 'http://www.reuters.com',
        'The American Conservative': 'http://www.theamericanconservative.com/',
        'The Hill': 'http://thehill.com',
        'The New York Times': 'http://www.nytimes.com',
        'The Wall Street Journal': 'http://www.wsj.com',
        'The Washington Post': 'https://www.washingtonpost.com',
        'The Weekly Standard': "https://www.weeklystandard.com/",
        'Washington Examiner': "https://www.washingtonexaminer.com/",
        'NPR': "https://www.npr.org/",
        'The Guardian': "https://www.theguardian.com/",
        'Occupy Democrats': "http://occupydemocrats.com/",
        'Daily Kos': "https://www.dailykos.com/",
        'The Atlantic': "https://www.theatlantic.com/",
        'Vox': "https://www.vox.com/",
        'Huffpost': "https://www.huffingtonpost.com/",
    }

    bias_score = {
        'hyper_left': -3,
        'left': -2,
        'center_left': -1,
        'center': 0,
        'center_right': 1,
        'right': 2,
        'hyper_right': 3,
    }

    # all potential sources
    bias_sources_map = {
        'hyper_left':    ['Occupy Democrats','Daily Kos'],
        'left':          ['MSNBC','Buzzfeed','The Atlantic','Vox','Huffpost'],
        'center_left':   ['The Guardian', 'Politico','The Washington Post','The New York Times','CNN',],
        'center':        ['Reuters','Associated Press', 'NPR'],
        'center_right':  ['The Wall Street Journal','The Hill'],
        'right':         ['National Review', 'New York Post','The Weekly Standard','Examiner', 'Washington Examiner'],
        'hyper_right':   ['Fox News','Breitbart News','The American Conservative'],
    }

    def __init__(self):
        self.sources_bias_map = {}
        for bias, source_list in self.bias_sources_map.items():
            self.sources_bias_map.update({s: bias for s in source_list})



if __name__ == '__main__':
    print(Sources().sources_bias_map)
