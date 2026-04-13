import markdown


def md_to_html(md_file):

    with open(md_file, "r", encoding="utf8") as f:
        text = f.read()

    html = markdown.markdown(text)

    html_file = md_file.replace(".md", ".html")

    with open(html_file, "w", encoding="utf8") as f:
        f.write(html)

    return html_file