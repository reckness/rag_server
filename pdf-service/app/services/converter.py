import os
from app.services.libreoffice import libreoffice_convert
from app.services.markdown_converter import md_to_html


def convert_to_pdf(file):

    ext = file.split(".")[-1].lower()

    if ext == "md":
        file = md_to_html(file)

    pdf = libreoffice_convert(file)

    return pdf