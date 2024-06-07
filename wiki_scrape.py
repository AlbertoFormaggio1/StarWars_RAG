# The plan is that we will process:
# 1) The films of the normal saga from https://starwars.fandom.com/wiki/Star_Wars_saga
# 2) The films from the categories
# 3) Get the pages of the characters appearing in the films or movies

import requests
from bs4 import BeautifulSoup
import re
import os

film_link = "https://en.wikipedia.org/wiki/List_of_Star_Wars_films"
series_link = "https://en.wikipedia.org/wiki/List_of_Star_Wars_television_series"
characters_link = "https://en.wikipedia.org/wiki/List_of_Star_Wars_characters"


def convert_page_to_text(html_tag, title_tags, keep_headers, headers, page_title):
    """

    :param html_tag:
    :param title_tags:
    :param keep_headers: True if you want to keep the paragraph, False if ignore the headers provided in the parameter headers
    :param headers:
    :return:
    """
    regex_headers = ("|".join(headers))
    # Unwrap meta tags
    meta_tags = html_tag.find_all("meta")
    for meta_tag in meta_tags:
        meta_tag.unwrap()

    full_page_text = ""
    last_tag_name = None
    ignore_paragraph = False

    full_page_text += f"<h1>{page_title}</h1>"

    for tag in html_tag.find_all("span", {"class": "IPA"}):
        tag.decompose()

    for child_tag in html_tag.children:
        if child_tag.name == "p":
            if not ignore_paragraph:
                # In the case the last tag was not a paragraph, add the start of a paragraph tag <p>
                if last_tag_name is None or last_tag_name != "p":
                    full_page_text += "<p>"

                full_page_text += child_tag.text
                last_tag_name = "p"
        else:
            # If now we're reading a tag that is not p and the last tag was p, close the tag.
            if last_tag_name is not None and last_tag_name == "p":
                full_page_text += "</p>"
            last_tag_name = child_tag.name

            # If the header of the paragraph is to be ignored, just skip it.
            if child_tag.name == "h2" and re.search(regex_headers, child_tag.text, flags=re.IGNORECASE):
                ignore_paragraph = not keep_headers
            elif child_tag.name == "h2":
                ignore_paragraph = keep_headers

            # Keep the header tag
            if not ignore_paragraph and child_tag.name in title_tags:
                full_page_text += f'<{child_tag.name}>{child_tag.text}</{child_tag.name}>'

    # Remove [] as citation to the notes
    full_page_text = re.sub("\[.+?\]", "", full_page_text)
    # Remove special &...; characters, special escape chars in HTML
    full_page_text = re.sub("&(.)+;", "", full_page_text)
    # Remove strange pronounciation of the characters
    full_page_text = re.sub("\((.)*Japanese(.)*\)", "", full_page_text)

    return full_page_text


def save_pages(web_pages_texts, file_names, directory):
    for file_name, text in zip(file_names, web_pages_texts):
        file_name = directory + file_name + ".html"

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(file_name, "w", encoding='utf-8') as html_file:
            print("saving file")
            html_file.write(text)

def wikipedia_extract_link(content_tag, link_name_regex):
    links = content_tag.find_all("a")
    link_dict = {tag.text: tag.get("href") for tag in links}

    compiled_wiki_regex = re.compile("wiki")
    remove_key = []

    for key, link in link_dict.items():
        if link is None or not re.search(compiled_wiki_regex, link):
            remove_key.append(key)

    for k in remove_key:
        del link_dict[k]

    relevant_links = []

    for key, value in link_dict.items():
        if re.search(link_name_regex, key):
            relevant_links.append(value)

    for idx, link in enumerate(relevant_links):
        relevant_links[idx] = re.sub("#(.+)$", "", link)

    # Remove duplicates
    relevant_links = list(set(relevant_links))

    for idx, link in enumerate(relevant_links):
        relevant_links[idx] = "https://en.wikipedia.org/" + link

    return relevant_links


def extract_page_content(link_list, keep_headers_bool, headers, title_tags):
    web_pages_texts = []
    file_names = []

    for link in link_list:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")
        div_content = soup.find("div", {"class": "mw-content-ltr mw-parser-output"})
        page_title = soup.find("h1", {"id": "firstHeading"}).text

        web_pages_texts.append(convert_page_to_text(div_content, title_tags, keep_headers_bool, headers, page_title))

        file_name = re.findall("/wiki/(.+)", link)[0]
        file_name = re.sub(":", "", file_name)
        file_names.append(file_name)

    return web_pages_texts, file_names


def export_movies():
    film_page = requests.get(film_link)
    soup_film = BeautifulSoup(film_page.content, "html.parser")

    div_content = soup_film.find("div", {"class": "mw-content-ltr mw-parser-output"})

    title_tags = ["h1", "h2", "h3", "h4", "h5"]
    ignore_headers = ["Reception", "Unproduced and abandoned projects", "Documentaries", "Notes", "See also",
                      "References", "External links"]

    full_page_text = convert_page_to_text(div_content, title_tags, False, ignore_headers, "List of Star Wars films")

    save_pages([full_page_text], ["Movies"], directory="./web_pages/")

    film_title_regex = re.compile("Episode|Rogue One|The Clone Wars|^Solo: A Star Wars Story$", flags=re.IGNORECASE)
    relevant_links = wikipedia_extract_link(div_content, film_title_regex)

    keep_headers = ["Plot", "Cast"]

    web_pages_texts, file_names = extract_page_content(relevant_links, True, keep_headers, title_tags)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")


def wikipedia_extract_character_link(html_tag):
    div_link_container = html_tag.find_all("div", {"role": "note"})

    links = []
    for div in div_link_container:
        link_to_resource = div.find("a")["href"]
        full_link = "https://en.wikipedia.org/" + link_to_resource
        links.append(full_link)

    return links


def export_characters():
    characters_page = requests.get(characters_link)
    soup_chars = BeautifulSoup(characters_page.content, "html.parser")

    div_content = soup_chars.find("div", {"class": "mw-content-ltr mw-parser-output"})

    title_tags = ["h1", "h2", "h3", "h4", "h5"]
    ignore_headers = ["References", "External Links"]

    full_page_text = convert_page_to_text(div_content, title_tags, False, ignore_headers, "List of Star Wars characters")

    save_pages([full_page_text], ["Characters"], directory="./web_pages/")

    relevant_links = wikipedia_extract_character_link(div_content)

    web_pages_texts, file_names = extract_page_content(relevant_links, False, [], title_tags)
    print(file_names)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")


def convert_page_series_to_text(html_tag, page_title):
    # Remove bad characters associated to pronounciation
    for tag in html_tag.find_all("span", {"class": "IPA"}):
        tag.decompose()

    full_page_text = ""

    full_page_text += f"<h1>{page_title}</h1>"
    full_page_text += "<p>"

    for child_tag in html_tag.children:
        print(child_tag.name)
        if child_tag.name == "h2":
            full_page_text += "</p>"
            break

        if child_tag.name == "p":
            full_page_text += child_tag.text

    episode_titles = html_tag.find_all("td", {"class": "summary"})
    episode_summaries = html_tag.find_all("td", {"class": "description"})

    for title, summary in zip(episode_titles, episode_summaries):
        title_tag = f"<title>{title.text}</title>"
        summary_tag = f"<p>{summary.text}</p>"
        full_page_text += title_tag + summary_tag

    # Remove [] as citation to the notes
    full_page_text = re.sub("\[.+?\]", "", full_page_text)
    # Remove special &...; characters, special escape chars in HTML
    full_page_text = re.sub("&(.)+;","", full_page_text)
    # Remove strange pronounciation of the characters
    full_page_text = re.sub("\((.)*Japanese(.)*\)", "", full_page_text)

    return full_page_text

def extract_series_content(links):
    pages_texts = []
    file_names = []

    for link in links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")

        div_content = soup.find("div", {"class": "mw-content-ltr mw-parser-output"})
        page_title = soup.find("h1", {"id": "firstHeading"}).text

        page_text = convert_page_series_to_text(div_content, page_title)
        pages_texts.append(page_text)

        file_name = re.findall("/wiki/(.+)", link)[0]
        file_name = re.sub(":", "", file_name)
        file_names.append(file_name)

    return pages_texts, file_names


def export_series():
    series_page = requests.get(series_link)
    soup_series = BeautifulSoup(series_page.content, "html.parser")

    div_content = soup_series.find("div", {"class": "mw-content-ltr mw-parser-output"})

    div_link_container = div_content.find_all("div", {"role": "note"})

    links = []
    for div in div_link_container:
        link_to_resource = div.find("a")

        if link_to_resource is None:
            continue

        link_to_resource = link_to_resource["href"]
        full_link = "https://en.wikipedia.org/" + link_to_resource
        links.append(full_link)

    web_pages_texts, file_names = extract_series_content(links)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")