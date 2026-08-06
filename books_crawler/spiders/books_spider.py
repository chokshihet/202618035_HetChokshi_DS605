import scrapy

class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]
    
    items_scraped = 0
    max_items = 100  # Scrapes exactly 100 books across 5 catalog pages

    def parse(self, response):
        book_links = response.css("article.product_pod h3 a::attr(href)").getall()
        for link in book_links:
            if self.items_scraped >= self.max_items:
                return
            full_url = response.urljoin(link)
            yield scrapy.Request(url=full_url, callback=self.parse_book_details)

        # Pagination: Stop following 'Next' once 100 items are reached
        next_page = response.css("li.next a::attr(href)").get()
        if next_page and self.items_scraped < self.max_items:
            next_page_url = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_url, callback=self.parse)

    def parse_book_details(self, response):
        if self.items_scraped >= self.max_items:
            return

        self.items_scraped += 1

        rating_classes = response.css("p.star-rating::attr(class)").get("")
        rating_raw = rating_classes.replace("star-rating", "").strip()

        description = response.xpath(
            "//div[@id='product_description']/following-sibling::p/text()"
        ).get()

        yield {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.xpath("//ul[@class='breadcrumb']/li[3]/a/text()").get(),
            "price": response.css("div.product_main p.price_color::text").get(),
            "rating": rating_raw,
            "availability": response.css("div.product_main p.instock.availability::text").getall(),
            "product_description": description if description else "",
            "UPC": response.xpath("//table//tr[th='UPC']/td/text()").get(),
            "number_of_reviews": response.xpath("//table//tr[th='Number of reviews']/td/text()").get(),
            "product_url": response.url
        }