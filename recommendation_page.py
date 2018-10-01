import os
from flask import url_for
from urllib.parse import urlparse
from sources import Sources

flatten = lambda l: [item for sublist in l for item in sublist] # TODO

class RecommendationPage:

    def __init__(self, app):
        self.app = app
        self.sources = Sources()

    def resolve_source_name(self, url):
        parsed_uri = urlparse(url)
        return self.sources.url_names.get(parsed_uri.netloc, 'Unknown source')

    def resolve_source_id(self, name):
        return self.sources.resolve_source_id(name)

    def resolve_bias_display(self, source_code):
        bias = self.sources.sources_bias_map[source_code]
        return self.sources.bias_display_names[bias]

    def build_recommendations_list(self, df, subset, top_n):
        df = df.head(top_n)

        return [row.to_dict() for i, row in df.iterrows()]

    def append_bias(self, rec_df):
        """
        For diplaying the bias information for each recommended article
        """
        bias_map = self.sources.sources_bias_map
        rec_df['publication'] = rec_df['source'].map(lambda x: x['name'])
        rec_df['bias_code'] = rec_df['source'].map(lambda x: (bias_map.get(x['id'])))
        rec_df['icon_url'] = rec_df['source'].map(lambda x: self.resolve_img_url(x['id']))
        # bias = rec_df['publication'].map(lambda x: bias_map.get(x, bias_map.get('The ' + x, bias_map.get(x + ' News'))))
        rec_df['bias_label'] = rec_df['bias_code'].map(lambda b: self.sources.bias_display_names.get(b))
        # df['bias_score'] = bias.map(lambda b: self.sources.bias_score.get(b))
        return rec_df

    def resolve_bias_code(self, source_code):
        return self.sources.sources_bias_map[source_code]

    def resolve_valid_biases(self, input_bias):
        biases = list(self.sources.bias_sources_map.keys())
        left = biases[:3]
        right = biases[4:]
        extreme = biases[0:1] + biases[-1:]

        if input_bias in left:
            valid = right + [biases[3]]
        elif input_bias in right:
            valid = left + [biases[3]]
        else: # source is relatively unbiased- hit em with the crazy
            valid = extreme

        if input_bias in extreme:
            valid.append(biases[3]) # append center

        return valid# + [None] # include None case for if no bias was determined

    def resolve_img_url(self, source_id):
        jpg = 'imgs/{}.jpg'.format(source_id)
        # png = 'imgs/{}.png'.format(source_id)
        jpg_file = url_for('static', filename=jpg)

        return jpg_file

        # TODO support png or whatever if that's what we have
        # png = 'imgs/{}.png'.format(source_id)
        # if os.path.isfile(os.path.join(self.app.instance_path, jpg_file)):
        #     print('is not file:', jpg_file)
        # else:
        #     print('going with file:', url_for('static', filename=png))
        #     return url_for('static', filename=png)

    def get_valid_sources(self, valid_biases):
        return flatten([self.sources.bias_source_id_map[b] for b in valid_biases])

if __name__ == "__main__":
    rec = RecommendationPage()
    bias_code = 'hyper_left'
    print(rec.resolve_valid_biases(bias_code))
