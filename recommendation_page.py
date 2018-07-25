import os
from flask import url_for
from urllib.parse import urlparse
from sources import Sources

class RecommendationPage:
    def __init__(self):
        self.sources = Sources()

    def resolve_source_name(self, url):
        parsed_uri = urlparse(url)
        return self.sources.url_names.get(parsed_uri.netloc, 'Unknown source')

    def resolve_source_id(self, name):
        return self.sources.resolve_source_id(name)

    def resolve_bias_display(self, source_name):
        bias = self.sources.sources_bias_map[source_name]
        return self.sources.bias_display_names[bias]

    def build_recommendations_list(self, df, top_n):
        df = df[['title', 'publication', 'url', 'partial_content']].head(top_n)

        return [row.to_dict() for i, row in df.iterrows()]

    def append_bias(self, df):
        bias_map = self.sources.sources_bias_map
        bias = df['publication'].map(lambda x: bias_map.get(x, bias_map.get('The ' + x)))
        df['bias_label'] = bias.map(lambda b: self.sources.bias_display_names.get(b))
        df['bias_score'] = bias.map(lambda b: self.sources.bias_score.get(b))
        return df

    def resolve_bias_code(self, source_name):
        return self.sources.sources_bias_map[source_name]

    def resolve_valid_biases(self, input_bias):
        biases = list(self.sources.bias_sources_map.keys())
        print(input_bias)
        left = biases[:3]
        # print(left)
        right = biases[4:]
        # print(right)
        extreme = biases[0:1] + biases[-1:]
        # print(extreme)

        if input_bias in left:
            valid = right
        elif input_bias in right:
            valid = left
        else: # source is relatively unbiased- hit em with the crazy
            valid = extreme

        if input_bias in extreme:
            valid.append(biases[3]) # append center

        return valid + [None] # include None case for if no bias was determined

    def resolve_img_url(self, source_id):
        png = 'imgs/{}.png'.format(source_id)
        jpg = 'imgs/{}.jpg'.format(source_id)
        jpg_file = url_for('static', filename=jpg)
        if os.path.isfile(jpg_file):
            return jpg_file
        else:
            return url_for('static', filename=png)

if __name__ == "__main__":
    rec = RecommendationPage()
    bias_code = 'hyper_left'
    print(rec.resolve_valid_biases(bias_code))
