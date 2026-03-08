"""
Reddit Tool Handler — Search Reddit for community discussions and opinions.

Uses Reddit's public JSON API (appending .json to URLs). No API key required.
Rate-limited by using a descriptive User-Agent per Reddit's API guidelines.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

_REDDIT_BASE = "https://www.reddit.com"
_VALID_SORTS = {"relevance", "hot", "top", "new", "comments"}
_VALID_TIME_FILTERS = {"day", "week", "month", "year", "all"}
_USER_AGENT = "Chalie/1.0 (community data aggregator; contact: opensource)"


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """
    Search Reddit for posts matching a query, optionally within a specific subreddit.

    Args:
        topic: Conversation topic (passed by framework, may inform search context)
        params: Tool parameters from manifest:
            - query (str, required): Search terms
            - subreddit (str, optional): Target subreddit (without r/ prefix)
            - sort (str, optional): 'relevance', 'hot', 'top', 'new', 'comments'
            - time_filter (str, optional): 'day', 'week', 'month', 'year', 'all'
            - limit (int, optional): Number of posts (1-8, default 5)
            - include_comments (bool, optional): Fetch top comments per post (default True)
        config: Tool config from DB (unused — no API key needed)
        telemetry: Client telemetry (unused for Reddit)

    Returns:
        dict with keys: results (list), count (int), query (str), _meta (dict)
        Each result: {title, subreddit, author, score, upvote_ratio, num_comments,
                      url, permalink, selftext, created_utc, top_comments}
    """
    query = (params.get("query") or "").strip()
    if not query:
        return {"results": [], "count": 0, "query": "", "_meta": {}}

    limit = max(1, min(8, int(params.get("limit") or 5)))
    subreddit = (params.get("subreddit") or "").strip().lstrip("/").removeprefix("r/")

    sort = (params.get("sort") or "relevance").strip().lower()
    if sort not in _VALID_SORTS:
        sort = "relevance"

    time_filter = (params.get("time_filter") or "month").strip().lower()
    if time_filter not in _VALID_TIME_FILTERS:
        time_filter = "month"

    include_comments = params.get("include_comments", True)
    if isinstance(include_comments, str):
        include_comments = include_comments.lower() not in ("false", "0", "no")

    t0 = time.time()
    posts, error = _search_reddit(query, subreddit, sort, time_filter, limit)

    if error and not posts:
        fetch_latency_ms = int((time.time() - t0) * 1000)
        logger.error(
            '{"event":"reddit_fetch_error","query":"%s","error":"%s","latency_ms":%d}',
            query, str(error)[:120], fetch_latency_ms,
        )
        return {
            "results": [], "count": 0, "query": query,
            "error": str(error)[:200],
            "_meta": {"fetch_latency_ms": fetch_latency_ms},
        }

    if include_comments and posts:
        _attach_comments(posts)

    fetch_latency_ms = int((time.time() - t0) * 1000)

    logger.info(
        '{"event":"reddit_search_ok","query":"%s","count":%d,"sort":"%s",'
        '"subreddit":"%s","latency_ms":%d}',
        query, len(posts), sort, subreddit or "all", fetch_latency_ms,
    )

    return {
        "results": posts,
        "count": len(posts),
        "query": query,
        "_meta": {
            "fetch_latency_ms": fetch_latency_ms,
            "post_count_raw": len(posts),
            "subreddit_searched": subreddit or "all",
            "sort": sort,
            "time_filter": time_filter,
        },
    }


# ── Reddit API fetch ───────────────────────────────────────────────────────────

def _search_reddit(query: str, subreddit: str, sort: str, time_filter: str, limit: int) -> tuple:
    """Call Reddit's public search JSON endpoint. Returns (posts, error)."""
    if subreddit:
        url = f"{_REDDIT_BASE}/r/{subreddit}/search.json"
        api_params = {
            "q": query,
            "restrict_sr": "on",
            "sort": sort,
            "t": time_filter,
            "limit": limit,
        }
    else:
        url = f"{_REDDIT_BASE}/search.json"
        api_params = {
            "q": query,
            "sort": sort,
            "t": time_filter,
            "limit": limit,
        }

    try:
        resp = requests.get(
            url,
            params=api_params,
            timeout=10,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [], e

    children = (data.get("data") or {}).get("children") or []
    posts = []
    seen_ids = set()

    for child in children:
        if child.get("kind") != "t3":
            continue
        post = child.get("data") or {}

        post_id = post.get("id", "")
        if not post_id or post_id in seen_ids:
            continue
        seen_ids.add(post_id)

        title = (post.get("title") or "").strip()
        if not title:
            continue

        selftext = (post.get("selftext") or "").strip()
        if selftext in ("[deleted]", "[removed]"):
            selftext = ""
        if len(selftext) > 400:
            selftext = selftext[:400] + "\u2026"

        permalink = post.get("permalink", "")
        post_url = f"{_REDDIT_BASE}{permalink}" if permalink else ""

        # Use external link URL for link posts; fall back to Reddit permalink for text posts
        external_url = post.get("url", "")
        if external_url and external_url.startswith("/r/"):
            # Relative Reddit URL — make absolute
            external_url = f"{_REDDIT_BASE}{external_url}"
        # If the external URL is just the Reddit permalink itself, clear it
        if external_url == post_url:
            external_url = ""

        posts.append({
            "id": post_id,
            "title": title,
            "subreddit": post.get("subreddit", ""),
            "author": post.get("author", ""),
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", 0.0),
            "num_comments": post.get("num_comments", 0),
            "url": external_url,
            "permalink": post_url,
            "selftext": selftext,
            "created_utc": post.get("created_utc", 0),
            "top_comments": [],
        })

    return posts, None


def _attach_comments(posts: list) -> None:
    """Fetch top comments for each post in-place. Best-effort; silently skips on error."""
    for post in posts:
        post_id = post.get("id")
        subreddit = post.get("subreddit")
        if not post_id or not subreddit:
            continue
        try:
            url = f"{_REDDIT_BASE}/r/{subreddit}/comments/{post_id}.json"
            resp = requests.get(
                url,
                params={"limit": 5, "sort": "top", "depth": 1},
                timeout=6,
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()

            # data[1] = comment listing
            if not isinstance(data, list) or len(data) < 2:
                continue

            comment_children = (data[1].get("data") or {}).get("children") or []
            comments = []
            for child in comment_children:
                if child.get("kind") != "t1":
                    continue
                cd = child.get("data") or {}
                body = (cd.get("body") or "").strip()
                if not body or body in ("[deleted]", "[removed]"):
                    continue
                if len(body) > 280:
                    body = body[:280] + "\u2026"
                comments.append({
                    "author": cd.get("author", ""),
                    "body": body,
                    "score": cd.get("score", 0),
                })
                if len(comments) >= 3:
                    break

            post["top_comments"] = comments

        except Exception as e:
            logger.debug(
                '{"event":"reddit_comments_fetch_failed","post_id":"%s","error":"%s"}',
                post_id, str(e)[:80],
            )
