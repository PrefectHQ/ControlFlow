import httpx
from markdownify import markdownify as md


def get_url(
    url: str, clean: bool = True, clean_images: bool = True, clean_links: bool = False
) -> str:
    """
    Make a GET request to the given URL, exactly as provided,
    and return the response text.

    If clean is True, the response text is converted from HTML to a
    Markdown-like format that removes extraneous tags. The `clean_images`
    (removes image tags) and `clean_links` (removes link tags) parameters can be
    used to further clean the text.

    Raises an exception if the response status code is not 2xx.
    """
    response = httpx.get(url)
    response.raise_for_status()
    if clean:
        strip = []
        if clean_images:
            strip.append("img")
        if clean_links:
            strip.append("a")
        return md(response.text, strip=strip)
    return response.text
