import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://www.agnoshealth.com"

def get_forum_links():
    """ดึงลิงก์กระทู้ทั้งหมด"""
    url = f"{BASE_URL}/forums"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    links = []
    for a_tag in soup.select("a[href^='/forums/']"):
        href = a_tag.get("href")
        if href and href not in links and "/forums/" in href:
            links.append(BASE_URL + href)
    return links

def get_post_content(url):
    """ดึง title และเนื้อหา"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    content_divs = soup.select("div")
    all_text = []
    for div in content_divs:
        text = div.get_text(" ", strip=True)
        if text and len(text) > 10:
            all_text.append(text)
    content = " ".join(all_text)

    return {
        "url": url,
        "title": title,
        "content": content
    }

def main():
    print("กำลังดึงลิงก์กระทู้")
    forum_links = get_forum_links()
    print(f" พบ {len(forum_links)} กระทู้")

    data = []
    for idx, link in enumerate(forum_links, start=1):
        print(f"[{idx}/{len(forum_links)}] ดึงข้อมูลจาก: {link}")
        try:
            post_data = get_post_content(link)
            data.append(post_data)
        except Exception as e:
            print(f"Error: {e}")

    # บันทึกเป็น JSON
    with open("agnos_forum_posts.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("บันทึกไฟล์เรียบร้อย")

if __name__ == "__main__":
    main()
