import json

import pandas as pd
import requests

from dagster import Config, MaterializeResult, MetadataValue, asset, Definitions, define_asset_job, AssetSelection, ScheduleDefinition, DefaultScheduleStatus


class HNStoriesConfig(Config):
    top_stories_limit: int = 10
    hn_top_story_ids_path: str = "hackernews_top_story_ids.json"
    hn_top_stories_path: str = "hackernews_top_stories.csv"


@asset(group_name='data_loading')
def data_fethcing(config: HNStoriesConfig):
    pass


@asset(deps=[data_fethcing],group_name='test_gr_name')
def hackernews_top_stories(config: HNStoriesConfig) -> MaterializeResult:
    """Get items based on story ids from the HackerNews items endpoint."""
    with open(config.hn_top_story_ids_path, "r") as f:
        hackernews_top_story_ids = json.load(f)

    results = []
    for item_id in hackernews_top_story_ids:
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        ).json()
        results.append(item)

    df = pd.DataFrame(results)
    df.to_csv(config.hn_top_stories_path)

    print("working here afhnowainhfogwein")

    return MaterializeResult(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(str(df[["title", "by", "url"]].to_markdown())),
        }
    )


ecommerce_job = define_asset_job(
    "ecommerce_job", AssetSelection.groups("test_gr_name")
)


ecommerce_schedule = ScheduleDefinition(
    job=ecommerce_job,
    cron_schedule="30 5 * * 1-5",
    default_status=DefaultScheduleStatus.RUNNING,
)


defs = Definitions(
    assets=[hackernews_top_story_ids, hackernews_top_stories],
    jobs=[ecommerce_job],
    schedules=[ecommerce_schedule],
)
