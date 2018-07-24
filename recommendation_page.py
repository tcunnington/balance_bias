from urllib.parse import urlparse
from sources import Sources

class RecommendationPage:
    def __init__(self):
        self.sources = Sources()

    def resolve_source_name(self, url):
        parsed_uri = urlparse(url)
        return self.sources.url_names.get(parsed_uri.netloc, 'Unknown source')


    def resolve_source_id(self, url):
        name = self.resolve_source_name(url)
        return self.sources.resolve_source_id(name)

if __name__ == "__main__":
    rec = RecommendationPage()
