

class Sources:

    name_urls = {
        'Associated Press': 'apnews.com',
        'Breitbart News': 'www.breitbart.com',
        'Buzzfeed': 'www.buzzfeed.com',
        'CNN': 'us.cnn.com',
        'Fox News': 'www.foxnews.com',
        'MSNBC': 'www.msnbc.com',
        'National Review': 'www.nationalreview.com',
        'Politico': 'www.politico.com',
        'Reuters': 'www.reuters.com',
        'The American Conservative': 'www.theamericanconservative.com',
        'The Hill': 'thehill.com',
        'The New York Times': 'www.nytimes.com',
        'The Wall Street Journal': 'www.wsj.com',
        'The Washington Post': 'www.washingtonpost.com',
        'The Weekly Standard': "www.weeklystandard.com",
        'Washington Examiner': "www.washingtonexaminer.com",
        'NPR': "www.npr.org/",
        'The Guardian': "www.theguardian.com",
        'Occupy Democrats': "occupydemocrats.com",
        'Daily Kos': "www.dailykos.com",
        'The Atlantic': "www.theatlantic.com",
        'Vox': "www.vox.com/",
        'Huffpost': "www.huffingtonpost.com",
    }

    bias_score = {
        'hyper-left': -3,
        'left': -2,
        'center-left': -1,
        'center': 0,
        'center-right': 1,
        'right': 2,
        'hyper-right': 3,
    }

    # all potential sources
    bias_sources_map = {
        'hyper-left':    ['Occupy Democrats','Daily Kos'],
        'left':          ['MSNBC','Buzzfeed News','The Atlantic','Vox','Huffpost','Talking Points Memo'],
        'center-left':   ['The Guardian', 'Politico','The Washington Post','The New York Times','CNN', 'Business Insider'],
        'center':        ['Reuters','Associated Press', 'NPR'],
        'center-right':  ['The Wall Street Journal','The Hill'],
        'right':         ['National Review', 'New York Post','The Weekly Standard','Examiner', 'Washington Examiner'],
        'hyper-right':   ['Fox News','Breitbart News','The American Conservative'],
    }

    bias_display_names = {
        'hyper-left': 'Far left',
        'left': 'Left',
        'center-left': 'Center-left',
        'center': 'Least Biased',
        'center-right': 'Center-right',
        'right': 'Right',
        'hyper-right': 'Far right',
        'unknown': 'Unknown Source'
    }

    def __init__(self):
        self.sources_bias_map = {}
        for bias, source_list in self.bias_sources_map.items():
            self.sources_bias_map.update({s: bias for s in source_list})

        self.url_names = {url:name for name,url in self.name_urls.items()}

    @staticmethod
    def resolve_source_id(source_name):
        return source_name.lower().replace(' ', '-')


if __name__ == '__main__':
    s = Sources()
    print(s.sources_bias_map)
    print([s.resolve_source_id(name) for name in s.name_urls.keys()])
