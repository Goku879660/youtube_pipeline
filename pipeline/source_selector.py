from __future__ import annotations


OBJECT_TYPES = {
    "car", "automobile", "vehicle", "truck", "tank", "animal", "wolf", "lion", "tiger",
    "horse", "bird", "ship", "boat", "aircraft", "plane", "train", "machine", "artifact",
}


def select_source_order(scene_analysis: dict) -> list[str]:
    scene_type = scene_analysis.get("scene_type", "general")
    objects = {item.lower() for item in scene_analysis.get("objects", [])}
    entities = scene_analysis.get("entities", [])
    years = scene_analysis.get("years", [])

    if objects & OBJECT_TYPES:
        return ["open_images", "laion", "common_crawl"]
    if scene_type == "historical" or years or entities:
        return ["common_crawl", "laion", "open_images"]
    return ["laion", "open_images", "common_crawl"]
