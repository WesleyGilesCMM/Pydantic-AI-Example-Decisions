import datetime
from .embeddings import embed_text


def utc_now():
    return datetime.datetime.now(datetime.UTC)
