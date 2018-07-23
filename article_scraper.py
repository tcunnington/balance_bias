from retrying import retry
from newspaper import Article


class ArticleScraper:

    @staticmethod
    @retry(wait_fixed=100, stop_max_attempt_number=2)
    def scrape(url):

        article = Article(url)
        article.download()
        article.parse()

        return article.title, article.text