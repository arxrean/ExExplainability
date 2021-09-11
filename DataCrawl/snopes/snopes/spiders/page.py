import re
import pdb
import scrapy
import pandas as pd


class ArticleSpider(scrapy.Spider):
    def __init__(self):
        self.n = 0
        self.csv = pd.read_csv('./metadata.csv')
        self.csv['content_img_url'] = ''
        self.csv['explain'] = ''
        self.csv['claim'] = ''
        self.d = pd.read_csv('./metadata.csv', usecols=['title', 'link'])
        self.start_urls = [self.d.iat[self.n, 1]]

    name = 'article'

    def parse(self, response):
        CLAIM_SELECTOR = '.claim-wrapper'
        CONTENT_SELECTOR = '.content-wrapper.card'

        claim = response.css(CLAIM_SELECTOR).css('p ::text').extract_first()
        content_img = ' '.join(response.css(CONTENT_SELECTOR).css('img::attr(src)').extract())
        content_explain = ' '.join(response.css(CONTENT_SELECTOR).css('::text').extract()).encode('ascii', errors='ignore').decode('utf-8')
        content_explain = ' '.join([c for c in content_explain.split() if not c.isspace()])

        self.csv.at[self.n, 'claim'] = '' if claim is None else claim.strip()
        self.csv.at[self.n, 'content_img_url'] = content_img
        self.csv.at[self.n, 'explain'] = content_explain

        self.n += 1
        try:
            next_page = self.d.iat[self.n, 1]
        except:
            self.csv.to_csv('./article.csv', index=False)
            print('done.')
            return

        print(next_page)

        # print(next_page)
        if next_page:
            # normal page
            yield scrapy.Request(
                response.urljoin(next_page),
                callback=self.parse,
                dont_filter=True
            )
