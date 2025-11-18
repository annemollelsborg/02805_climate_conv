import json
import networkx as nx

def extract_tweet(t):
    entities = t.get("entities", {}) or {}

    # --- hashtags ---
    hashtags = [h.get("tag") for h in entities.get("hashtags", [])]

    # --- mentions ---
    raw_mentions = entities.get("mentions", [])
    mention_usernames = [m.get("username") for m in raw_mentions]
    mention_ids = [m.get("id") for m in raw_mentions]

    # node IDs for networkx: prefer id, otherwise username
    mention_nodes = [
        m.get("id") if m.get("id") is not None else m.get("username")
        for m in raw_mentions
    ]

    # --- urls ---
    urls = [
        u.get("expanded_url", u.get("url"))
        for u in entities.get("urls", [])
    ]

    # --- public metrics ---
    public = t.get("public_metrics", {}) or {}

    return {
        "id": t.get("id"),
        "author_id": t.get("author_id"),
        "created_at": t.get("created_at"),
        "text": t.get("text"),
        "lang": t.get("lang"),

        # public metrics
        "likes": public.get("like_count"),
        "replies": public.get("reply_count"),
        "retweets": public.get("retweet_count"),
        "quotes": public.get("quote_count"),
        "impressions": public.get("impression_count"),

        # extracted lists
        "hashtags": hashtags,
        "mention_usernames": mention_usernames,
        "mention_ids": mention_ids,
        "mention_nodes": mention_nodes,  # <- use this for graph
        "urls": urls,
    }