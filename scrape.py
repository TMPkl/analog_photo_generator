import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random

START_URL = "https://www.lomography.com/films/871920737-kodak-gold-200-35mm/photos/28766460?order=recent"
OUTPUT_DIR = "./images"

# Lista różnych User-Agent do rotacji
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_headers():
    """Losuje User-Agent z listy"""
    return {"User-Agent": random.choice(USER_AGENTS)}

def get_main_asset_image_url(asset_div, page_url):
    """Pobiera główny obraz z div.asset"""
    img_tag = asset_div.find("img")
    if img_tag:
        for attr in ["src", "data-src", "data-srcset"]:
            val = img_tag.get(attr)
            if val:
                if attr == "data-srcset":
                    parts = val.split(",")
                    val = parts[-1].split()[0].strip()
                return urljoin(page_url, val)
    
    picture = asset_div.find("picture")
    if picture:
        source = picture.find("source")
        if source and source.get("srcset"):
            val = source["srcset"].split(",")[-1].split()[0].strip()
            return urljoin(page_url, val)
    
    return None

def download_image(img_url, counter):
    """Pobiera obraz i zapisuje go numerując od counter"""
    filename = f"{counter}.jpg"
    out_path = os.path.join(OUTPUT_DIR, filename)
    try:
        resp = requests.get(img_url, headers=get_headers(), timeout=15)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"✅ Pobranie: {filename}")
    except Exception as e:
        print(f"❌ Błąd pobierania {img_url}: {e}")

def main():
    current_url = START_URL
    page_count = 1
    img_counter = 1

    while current_url:
        print(f"\nPobieram stronę {page_count}: {current_url}")
        try:
            resp = requests.get(current_url, headers=get_headers(), timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"❌ Nie udało się pobrać strony {current_url}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        asset_div = soup.find("div", class_="asset")
        if asset_div:
            img_url = get_main_asset_image_url(asset_div, current_url)
            if img_url:
                download_image(img_url, img_counter)
                img_counter += 1
                # Co 10 zdjęć: krótka pauza
                if img_counter % 10 == 1:  # po pobraniu 10, 20, 30 ...
                    sleep_time = random.randint(3, 6)
                    print(f"⏸ Pauza {sleep_time}s i zmiana User-Agent...")
                    time.sleep(sleep_time)
            else:
                print("⚠️ Nie udało się wyciągnąć URL głównego obrazka z <div class='asset'>")
        else:
            print("⚠️ Nie znaleziono <div class='asset'> na tej stronie")

        # Link do następnej strony
        next_link = soup.find("a", rel="next")
        if next_link and next_link.get("href"):
            current_url = urljoin(current_url, next_link["href"])
            page_count += 1
        else:
            print("\n✅ Brak kolejnej strony. Koniec.")
            break

if __name__ == "__main__":
    main()
