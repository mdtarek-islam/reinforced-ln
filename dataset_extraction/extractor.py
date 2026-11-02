"""Download the MyAnimeList light novel ranking and save it as CSV.

Set the MAL_CLIENT_ID environment variable to your MyAnimeList API client ID.
"""
import csv
import os

import requests

REQUEST_TIMEOUT_SECONDS = 30


def node_to_row(manga):
    """Convert one ranking ``node`` from the API into a CSV row."""
    genres = ", ".join([genre["name"] for genre in manga.get("genres", [])])
    picture = manga.get("main_picture") or {}
    synopsis = manga.get("synopsis") or "N/A"
    return [
        manga.get("id", "N/A"),
        manga.get("title", "N/A"),
        picture.get("medium", "N/A"),
        picture.get("large", "N/A"),
        manga.get("mean", "N/A"),
        manga.get("rank", "N/A"),
        manga.get("popularity", "N/A"),
        genres,
        synopsis.replace("\n", " ")
    ]


def fetch_light_novels(client_id=None):
    """Page through the ranking endpoint and return all rows."""
    client_id = client_id or os.environ.get("MAL_CLIENT_ID")
    if not client_id:
        raise RuntimeError("Set MAL_CLIENT_ID to your MyAnimeList API client ID.")
    base_url = "https://api.myanimelist.net/v2/manga/ranking"
    headers = {"X-MAL-CLIENT-ID": client_id}
    offset = 0
    limit = 500  # Max limit per request
    all_data = []
    
    while True:
        params = {
            "ranking_type": "lightnovels",
            "fields": "id,title,mean,rank,popularity,genres,synopsis,main_picture",
            "limit": limit,
            "offset": offset,
            "nsfw": "true"
        }
        
        response = requests.get(base_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break
        
        data = response.json()
        
        if "data" not in data or not data["data"]:
            break  # Exit if no more data available
        
        for item in data["data"]:
            all_data.append(node_to_row(item["node"]))

        offset += limit
    
    return all_data

def save_to_csv(data, filename="light_novels.csv"):
    """Write rows to ``filename`` with the dataset header."""
    headers = ["ID", "Title", "Medium_Picture", "Large_Picture", "Mean", "Rank", "Popularity", "Genres", "Synopsis"]
    
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    light_novels = fetch_light_novels()
    if light_novels:
        save_to_csv(light_novels)
    else:
        print("No data retrieved.")
