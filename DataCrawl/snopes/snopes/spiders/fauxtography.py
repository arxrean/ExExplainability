import re
import pdb
import scrapy


class FauxtographySpider(scrapy.Spider):
    def __init__(self):
        self.n = 0
        self.flag = 1

    name = 'fauxtography'
    start_urls = ['https://www.snopes.com/fact-check/category/photos/']

    def parse(self, response):
        SET_SELECTOR = '.media-list'

        for fauxset in response.css(SET_SELECTOR).css('article '):
            self.n += 1
            TITLE_SELECTOR = '.title ::text'
            SUB_TITLE_SELECTOR = '.subtitle ::text'
            IMAGE_SELECTOR = '.featured-media img::attr(src)'

            temp = fauxset.extract()
            childurl = re.findall(r'href="(.*?)"', temp)[0]
            truth = "".join(re.findall(r'fact_check_rating-(.*?) |"', temp))
            image_url = fauxset.css(IMAGE_SELECTOR).extract()[0]
            title = self.extract_title(fauxset, TITLE_SELECTOR)
            subtitle = self.extract_title(fauxset, SUB_TITLE_SELECTOR)

            yield {
                'count': self.n,
                'title': title,
                'subtitle': subtitle,
                'link': childurl,
                'cover_image_url': image_url,
                'ground_truth': truth.replace('"', '')
            }

        NEXT_PAGE_SELECTOR = '.btn-group a::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()
        if next_page:
            # handle last page
            if len(next_page) < 2 and self.flag == 0:
                return
            # first page
            elif len(next_page) < 2 and self.flag == 1:
                print(next_page[0])
                self.flag = 0
                yield scrapy.Request(
                    response.urljoin(next_page[0]),
                    callback=self.parse,
                    dont_filter=True
                )

            else:
                # normal page
                print(next_page[1])
                yield scrapy.Request(
                    response.urljoin(next_page[1]),
                    callback=self.parse,
                    dont_filter=True
                )

    def extract_title(self, body, SELECTOR):
        title = body.css(SELECTOR).extract()
        if len(title) == 1:
            return title[0]
        elif len(title) > 1:
            return '----'.join(title)
        else:
            return ''
