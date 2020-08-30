# TASK - given a SPRINGER book ID (ISBN 13 , ISBN 10), like - 978-3-540-32692-2
# or an e-ISBN Like - e-ISBN: 978-3-540-75892-1
# Search and WebScrape the name into a CSV file

"""
Initial thoughts - 

1/ Google Search - ISBN: 978-3-540-75892-1
2/ Automate with - requests and Selenium 
3/ Scrape content from within the Springer link - https://www.springer.com/gp/book/9783540758891
4/ Maybe directly hit the Spriger URL's by creating URL's by adding ISBN Strings - if we hit directly the 
# ISBN is an ISBN not the e-ISBN , like in this case its = ISBN: 978-3-540-75889-1 
"""

import requests

