"""Clean dataset preparation for Western/English NSFW classification.

Refined rules using all 207k exported torrents with file paths and extensions:
1. Exclude non-Western titles and non-Western file paths from Western training data
   (since non-Western scripts are handled deterministically by the language filter).
2. Use file paths & extensions:
   - Positive mining: adult keywords in name OR file paths, adult extensions/tags.
   - Negative mining: clean documents (.epub, .pdf, .mobi), clean audio (.flac, .mp3, .alac),
     software (.exe, .dll, .iso), clean movies/series with no adult terms in either name or file paths.
"""

import csv
import random
import re
from pathlib import Path

from bitmagnet_classify.langid import _is_cyrillic, _is_han, _is_hangul, _is_kana
from bitmagnet_classify.nsfw import (
    is_keyword_nsfw,
    is_mainstream_allowed,
)

# Explicit hard negatives (titles containing provocative words that are clean media)
HARD_NEGATIVES = [
    "Sex and the City S01E01 1080p BluRay x264",
    "Sex and the City Season 2 Complete 720p WEB-DL",
    "Sex Education S03E01 720p WEB-DL mkv",
    "Sex Education S01 Complete 1080p NF WEBRip",
    "The Sex Lives of College Girls S02E05 1080p",
    "Sex Pistols - Never Mind the Bollocks (1977) [FLAC]",
    "Masters of Sex S01 720p HDTV",
    "Masters of Sex S04 Complete 1080p BluRay",
    "Dan Carlin Hardcore History 65 Supernova in the East IV mp3",
    "Dan Carlin's Hardcore History - Blueprint for Armageddon",
    "The Big Bang Theory S10E01 720p HDTV",
    "The Big Bang Theory Season 1-12 Complete 1080p",
    "Kiss Kiss Bang Bang (2005) 1080p BluRay x264",
    "Bang Bang Baby S01 1080p Prime Video",
    "Chitty Chitty Bang Bang (1968) 1080p BluRay",
    "[HorribleSubs] Kill la Kill - 02 [720p].mkv",
    "[Erai-raws] Kill la Kill [01-24] [1080p]",
    "Kill Bill Vol 1 (2003) 1080p BluRay",
    "Kill Bill Vol 2 (2004) 1080p Remux",
    "To Kill a Mockingbird (1962) 1080p BluRay",
    "archive sea fighting in greece 1897 wealthxin zip",
    "Odna vojna 2009 avi",
    "War and Peace (1966) Criterion 1080p",
    "Band of Brothers (2001) 1080p BluRay",
    "Saving Private Ryan (1998) 2160p UHD HDR",
    "Fighting with My Family (2019) 1080p BluRay",
    "[Beatrice-Raws] Sword Art Online II 14.5 [BDRip 1920x1080 HEVC TrueHD]",
    "Love Happy (1949) Xvid 1cd - Marx Brothers, Vera Ellen -  Marilyn Monroe [DDR]",
    "OmenXiii - Grim [MQA]",
    "Lady_Bird_2017_WEB-DLRip_MVO_by_Dalemake",
    "202609032145jxyczycj577.zip",
    "VID_20200704_120150.mp4",
    "Roadside Research.24864017",
    "YP-EPUB-COMIC55.epub",
    "MIZD-553",
    "Reacher S02E04 1080p rus LostFilm TV mkv",
    "The Little Book of Common Sense Investing",
    "Project X (2012) Oliver Cooper 1080p H264 AC 3",
    "Stardust 2007 2160p UHD BluRay REMUX",
    "Thomas Thomassen SUbD v2.1.10 x64 Win",
    "Acrylic Texture Pack 3",
    "[Trix] The World Is Dancing S01 WEBRip 1080p AV1",
    "Kiki's Delivery Service [BDRip 720p]",
    "5-Minute Chair Yoga for Weight Loss",
    "BBC The Secret Life of Books Great Expectations",
    "Temptation Island S03E08 1080p WEB",
    "Aperture Kids And The Mysterious Neighbor (2021) [1080p]",
    "Horizon Forbidden West EUR",
    "Caught Stealing 2025 1080p HD x264",
    "The.Blacklist.5x22.Sutton.Ross.ITA-ENG.1080p.WEBMux.DD5.1.x264-NovaRip.mkv",
    "Nichijou [BD] [1080p]",
    "Lipps Inc - Funkytown .mpg",
    "Eloise at Christmastime (2003).avi",
    "Naked Gun 33",
    "Highway Patrol (4) (1955–1959), Broderick Crawford",
    "Comedy Club (2026.10.02) NNMClub.avi",
    "Armbian_26.2.1_Orangepi4-lts_trixie_current_6.18.8_minimal.img.xz",
    "The.Powerpuff.Girls.Special.Dance.Pantsed.1080p.WEB-DL.x264.AAC.mkv",
    "Adele - 21(Deluxe Edition)(By Alexis Castro & Agustin Mallea DJ's)",
    "K2 - The Ultimate High - DVDrip-XviD.avi",
    "[NonActionableFansubs]Kamen.Rider.Zero-One.Short.Anime.01-05.END.hardsubs.1080p.BD.x264.AAC",
    "Sin.[Il.Peccato].2019.1080p.Bluray.x264.AC3.HORiZON-ArtSubs.mkv",
    "Rahat Fateh Ali Khan [Vol.1] - [Bollywood] - (2011) - [Mp3] - [Kujvbnaday]",
    "Moana (2026) [1080p][Subtitulada][VerPeliculasOnline.org].mp4",
    "[Judas] ZENSHUU (Season 01) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]",
    "Real VR Fishing",
    "Yuri Kuma Arashi",
    "[MP3][D] Superchick",
    "Afrojack - GF Masters, Vol. 4 (2011) (AAC)",
    "Hetalia Axis Powers [DVDRip 576p]",
    "emilija.galotti.mp3",
    "Resident.Evil.2026.iTALiAN.MD.TS.HCSubs.XviD-WRS.avi",
    "X-Men [4K UHDremux][2160p][HDR][AC3 5.1-DTS 5.1Castellano DTS-MA 5.1-Ingles+Subs][ES-EN]",
    "French kiss (1995) BDRip-AVC [4xUkr.Eng Sub.Eng] [Hurtom].mkv",
    "Reina Roja [HDTV 720p][Cap.202]",
    "Cupids Proxy (2017) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "same_damn_time_big.mp4",
    "[ASO] Inu X Boku Secret Service [BD 720p AC3 Dual-Audio]",
    "Dirty_Filthy_Love",
    "Ana-ta-han (1953) [1080p] [YTS.AG]",
    "Cupids Proxy (2017) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "Moms Got A Date With A Vampire (2000) [720p] [BluRay] [YTS.MX]",
    "Lolita (1962) [BluRay] [1080p] [YTS.AM]",
    "Yakuza Ladies Decision (1998) [1080p] [WEBRip] [YTS.MX]",
    "Little Women (2019) [2160p] [4K] [WEB] [5.1] [YTS.MX]",
    "Chris Rock Bring The Pain (1996) [1080p] [WEBRip] [YTS.MX]",
    "Women (2021) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "Handsome Guys (2024) [1080p] [WEBRip] [x265] [10bit] [YTS.MX]",
    "An Affair (2018) [720p] [BluRay] [YTS.MX]",
    "Another Gay Movie (2006) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "Funny Girl (2018) [720p] [WEBRip] [YTS.MX]",
    "Cult Girls (2019) [720p] [WEBRip] [YTS.MX]",
    "Virgin Punk Clockwork Girl (2025) [1080p] [BluRay] [5.1] [YTS.BZ]",
    "Sweet Sugar (1972) [BluRay] [1080p] [YTS.LT]",
    "Temptation (1946) [1080p] [BluRay] [YTS.MX]",
    "Marci X (2003) [1080p] [WEBRip] [YTS.MX]",
    "Female Prisoner #701 Scorpion (1972) [1080p] [BluRay] [YTS.MX]",
    "Uptown Girls (2003) [1080p] [YTS.AG]",
    "Fanny (2013) [1080p] [BluRay] [5.1] [YTS.MX]",
    "The Girlfriend Experience (2009) [1080p] [BluRay] [5.1] [YTS.MX]",
    "Shes All That (1999) [1080p] [BluRay] [5.1] [YTS.MX]",
    "Girls Will Be Girls (2024) [1080p] [WEBRip] [x265] [10bit] [5.1] [YTS.MX]",
    "Holy Smoke (1999) [1080p] [BluRay] [YTS.MX]",
    "Vixen (1968) [UHD BLURAY] [1080p] [BluRay] [YTS.MX]",
    "An American Girl Chrissa Stands Strong (2009) [1080p] [WEBRip] [YTS.MX]",
    "Kinda Pregnant (2025) [2160p] [4K] [WEB] [5.1] [YTS.MX]",
    "32nd Annual Actor Awards (2026) [720p] [WEBRip] [YTS.BZ]",
    "Jackass 2.5 (2007) [480p] [DVDRip] [YTS.MX]",
    "Summer Temptations (1988) [GERMAN] [1080p] [BluRay] [YTS.MX]",
    "Female Fight Squad (2016) [BluRay] [1080p] [YTS.AM]",
    "Maleficent Mistress Of Evil (2019) [WEBRip] [1080p] [YTS.LT]",
    "True Love (1989) [1080p] [BluRay] [YTS.MX]",
    "Ruby Gillman Teenage Kraken (2023) [1080p] [WEBRip] [x265] [10bit] [5.1] [YTS.MX]",
    "Someone Like You (2024) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "Material Girls (2006) [1080p] [BluRay] [5.1] [YTS.MX]",
    "Fun Size (2012) [1080p] [BluRay] [5.1] [YTS.MX]",
    "Super Size Me 2 Holy Chicken (2017) [1080p] [WEBRip] [5.1] [YTS.MX]",
    "The Nun (2018) [BluRay] [1080p] [YTS.AM]",
    "Nanny McPhee (2005) [YTS.AG]",
    "Party Girl (1995) [1080p] [WEBRip] [YTS.MX]",
    "Little Women (1994) [720p] [BluRay] [YTS.MX]",
    "Alice Sweet Alice (1976) [BluRay] [1080p] [YTS.AM]",
    "Sushi Girl (2012) [1080p] [BluRay] [5.1] [YTS.MX]",
    "She Did That (2019) [1080p] [WEBRip] [YTS.MX]",
    "Kiss The Girls (1997) [1080p] [YTS.AG]",
    "[SubsPlease] Kamisama Kiss Specials (01-05) (720p) [Batch]",
    "Bang-On Balls - Chronicles [FitGirl Repack]",
    "[Erai-raws] Girls und Panzer Saishuushou - 03 [1080p][HEVC][7B98A7AB].mkv",
    "[HorribleSubs] Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai - 04 [1080p].mkv",
    "Black Knight (2001) [Ukr,Eng, Sub Eng] HDTVRip 720p [Hurtom] by Zeus62.mkv",
    "[Erai-raws] Girls und Panzer Saishuushou - 04 [480p HIDIVE WEB-DL AVC AAC][7C1E9669].mkv",
    "XO Kitty (S02) (2025) WEB-DL 1080p Ukr Eng [Hurtom]",
    "[SubsPlease] Summer Pockets - 23 (1080p) [E6A1F4A6].mkv",
    "[Erai-raws] Captain Tsubasa Season 2 - Junior Youth Hen - 19 [1080p][Multiple Subtitle][A2D2EF48].mkv",
    "[Erai-raws] Chanto Suenai Kyuuketsuki-chan - 04 [1080p CR WEBRip HEVC AAC][MultiSub][29A6EA6C].mkv",
    "[SubsPlease] Gachiakuta - 19 (720p) [9FFC4225].mkv",
    "True History of the Kelly Gang (2019) BDRip-AVC [Ukr,Eng] sub [Eng] [ Hurtom ].mkv",
    "[SubsPlease] Anne Shirley - 24 (1080p) [D53882BC].mkv",
    "[SubsPlease] Urusei Yatsura (2022) - 19 (1080p) [9A45405B].mkv",
    "[SubsPlease] Jigokuraku - 19 (1080p) [CE26A686].mkv",
    "[SubsPlease] Summer Pockets - 19 (1080p) [2F73278D].mkv",
    "[Judas] Isekai Shikkaku (No Longer Allowed in Another World) (Season 1) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]",
    "[Erai-raws] Summer Pockets - 17 [1080p CR WEBRip HEVC AAC][MultiSub][09FE7B4C].mkv",
    "[Judas] Seishun Buta Yarou wa Bunny Girl Senpai no Yume o Minai (Rascal Does Not Dream of Bunny Girl Senpai) (Season 02) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]",
    "[SubsPlease] Anne Shirley - 16 (1080p) [D9C2F23F].mkv",
    "CarX Street [FitGirl Repack]",
    "[The Criterion Collection] DVD Killer's Kiss",
]

_ADULT_LEAKAGE = re.compile(
    r"(?i)\b(?:anal|ass|bang|bangbros|bdsm|blowjob|bondage|boob|brazzers|cocks|creampie|cuckold|cum|deepthroat|dicks|dildo|erotic|evilangel|fansly|fc2|femboy|fetish|footjob|fuck|gangbang|gloryhole|handjob|hardcore|hentai|heyzo|incest|interracial|jav|kink|ladyboy|legalporno|lewd|masturbat|milf|naughtyamerica|nubile|onlyfans|orgasm|orgy|pegging|porn|pov|pthc|ptsc|pussy|realitykings|seduc|sex|shemale|slut|squirt|stepdaughter|stepmom|stepsister|strapon|stripper|succubus|swinger|taboo|teamskeet|threesome|tits|topless|transsexual|twistys|vibrator|wank|xxx|horny|cougar|bicurious|hottie|tomboy|uncensored|leaks|nude|camgirl|siterip|sweetheart|busty)\b"
)
_JAV_CODE = re.compile(r"\b[A-Za-z]{3,5}-\d{2,4}\b")


def is_non_western(text: str) -> bool:
    """Check if text contains non-Western scripts (Kana, Han, Hangul, Cyrillic)."""
    for ch in text:
        c = ord(ch)
        if _is_kana(c) or _is_han(c) or _is_hangul(c) or _is_cyrillic(c):
            return True
    return False


def extract_clean_dataset(
    csv_path: str = "scratch/titles_with_files.csv",
    num_positives: int = 15000,
    num_negatives: int = 15000,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Sample high-confidence positive and diverse clean negative Western torrent titles.

    Reads from titles_with_files.csv with columns: (name, sample_paths, extensions).
    """
    random.seed(seed)

    positives: list[tuple[str, int]] = []
    negatives: list[tuple[str, int]] = []

    path = Path(csv_path)
    if not path.is_file():
        # Fallback to titles.csv if titles_with_files.csv does not exist
        csv_path = "scratch/titles.csv"

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            title = row[0].strip()
            if not title or len(title) < 4:
                continue

            sample_paths = row[1].strip() if len(row) > 1 else ""

            # Skip if title or file paths contain non-Western scripts (Kana, Han, Hangul, Cyrillic)
            if is_non_western(title) or is_non_western(sample_paths):
                continue

            # Check positive (true adult release in title or sample files)
            has_adult_signal = (
                is_keyword_nsfw(title)
                or is_keyword_nsfw(sample_paths)
                or bool(_ADULT_LEAKAGE.search(sample_paths))
            )

            if has_adult_signal and not is_mainstream_allowed(title):
                if len(positives) < num_positives * 2:
                    positives.append((title, 1))
            elif (
                not is_keyword_nsfw(title)
                and not _ADULT_LEAKAGE.search(title)
                and not _JAV_CODE.search(title)
                and not _ADULT_LEAKAGE.search(sample_paths)
                and len(negatives) < num_negatives * 2
            ):
                negatives.append((title, 0))

    random.shuffle(positives)
    random.shuffle(negatives)

    selected_pos = positives[:num_positives]
    selected_neg = negatives[: num_negatives - len(HARD_NEGATIVES)]
    for hn in HARD_NEGATIVES:
        selected_neg.append((hn, 0))

    data = selected_pos + selected_neg
    random.shuffle(data)

    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data


if __name__ == "__main__":
    train, test = extract_clean_dataset()
    print(f"Train samples: {len(train):,} (Positives: {sum(y for _, y in train):,})")
    print(f"Test samples:  {len(test):,} (Positives: {sum(y for _, y in test):,})")
