import re

CLEAN_EXT_PATTERN = re.compile(
    r"\.(?:mp4|mkv|avi|wmv|mov|flv|iso|zip|rar|tar|gz|7z|epub|pdf|cbz|cbr|txt|nfo|sfv)$",
    re.IGNORECASE,
)

BRACKET_PATTERN = re.compile(r"\[([^\]]*)\]|\{([^\}]*)\}|\(([^\)]*)\)")

CLEAN_DELIMITER_PATTERN = re.compile(r"[\[\]{}()]")

CLEAN_SCENE_TAGS_PATTERN = re.compile(
    r"\b(?:\d{3,4}[pi]|uhd|4k|bluray|blu-ray|bdrip|brrip|dvdrip|dvd|web-?dl|webrip|web|hdtv|hdrip|remux|h\.?264|x\.?264|h\.?265|x\.?265|hevc|avc|av1|10bit|8bit|xvid|divx|aac|ac3|eac3|ddp?5\.1|dts(?:-hd)?|truehd|flac|mp3|opus|repack|proper|unrated|remastered|extended|director'?s cut|limited|internal|readnfo|rerip)\b",
    re.IGNORECASE,
)

NON_WESTERN_PATTERN = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002A6DF\uf900-\ufaff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u0400-\u052f]"
)


def _is_year(text: str) -> bool:
    return len(text) == 4 and text.isdigit() and text[:2] in {"19", "20"}


def _bracket_replacement(match: re.Match[str]) -> str:
    content = next(group for group in match.groups() if group is not None).strip()
    if _is_year(content) or is_keyword_nsfw(content):
        return f" {content} "
    return " "


def sanitize_title_for_model(text: str) -> str:
    if not text:
        return ""
    cleaned = CLEAN_EXT_PATTERN.sub(" ", text)
    cleaned = BRACKET_PATTERN.sub(_bracket_replacement, cleaned)
    cleaned = CLEAN_DELIMITER_PATTERN.sub(" ", cleaned)
    cleaned = CLEAN_SCENE_TAGS_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace(".", " ").replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


MAINSTREAM_ALLOWLIST = [
    r"sex\s+(?:and\s+the\s+city|education|life|lives\s+of\s+college\s+girls|pistols)",
    r"masters\s+of\s+sex",
    r"hardcore\s+history",
    r"(?:the\s+)?big\s+bang\s+theory",
    r"kiss\s+kiss\s+bang\s+bang",
    r"bang\s+bang\s+baby",
    r"chitty\s+chitty\s+bang\s+bang",
]

MAINSTREAM_PATTERN = re.compile(
    r"(?:^|[\W_])(?:" + "|".join(MAINSTREAM_ALLOWLIST) + r")(?:$|[\W_])",
    re.IGNORECASE,
)

KEYWORDS_XXX = [
    r"anal",
    r"ass(?:es)?",
    r"bang(?:bros|ing)",
    r"bdsm",
    r"belami",
    r"blacked(?:raw)?",
    r"brattysis",
    r"busty",
    r"blowjobs?",
    r"bondage",
    r"boob\w*",
    r"brazzers",
    r"caribbeancom",
    r"chaturbate",
    r"cocks?",
    r"creampie",
    r"crossdress\w*",
    r"cuck\w*",
    r"deeper",
    r"cum\w*",
    r"deepthroat",
    r"dicks?",
    r"digitaldesire",
    r"dildo",
    r"erito",
    r"erocomic",
    r"erotic\w*",
    r"evilangel",
    r"fansly\w*",
    r"fc2",
    r"femboy",
    r"familystrokes",
    r"fantia",
    r"fetish",
    r"fakehub",
    r"footjob",
    r"fuck\w*",
    r"gangbang",
    r"gloryhole\w*",
    r"handjob",
    r"hardcore",
    r"hegre",
    r"horny",
    r"hentai",
    r"heyzo",
    r"incest",
    r"interracial",
    r"jav",
    r"javhd",
    r"jizz\w*",
    r"just[\s_-]?for[\s_-]?fans?",
    r"kink\w*",
    r"ladyboy",
    r"legalporno",
    r"lesbian[sx]?",
    r"lewd\w*",
    r"masturbat\w*",
    r"manyvids",
    r"mature4k",
    r"mofos",
    r"momcomesfirst",
    r"milf\w*",
    r"naughtyamerica",
    r"netorare",
    r"nubile\w*",
    r"only[\s_-]?fans?",
    r"orgasm\w*",
    r"orgy",
    r"passionhd",
    r"panties",
    r"panty",
    r"pegging",
    r"playboy",
    r"playmate",
    r"realjamvr",
    r"porn\w*",
    r"pov",
    r"propertys?ex",
    r"prostitut\w*",
    r"puretaboo",
    r"pussy",
    r"realitykings",
    r"seduc\w*",
    r"sexart",
    r"sex\w*",
    r"sislovesme",
    r"shemale",
    r"sissy",
    r"slut\w*",
    r"squirt\w*",
    r"step[\s_-]?(?:daughter|mom|sister|son|dad|father|mother)",
    r"strapon",
    r"stripper",
    r"submissive",
    r"succubus",
    r"swinger",
    r"taboo",
    r"teamskeet",
    r"threesome",
    r"tits?",
    r"titties",
    r"tonguejob",
    r"topless",
    r"tokyo[\s_-]?hot",
    r"tushy(?:raw)?",
    r"transsexual",
    r"twistys",
    r"vibrator",
    r"vixen",
    r"wank\w*",
    r"x[-\s_]?art",
    r"xxx",
]

TOKEN_PATTERN = re.compile(
    r"(?:^|[\W_])(?:" + "|".join(KEYWORDS_XXX) + r")(?:$|[\W_])",
    re.IGNORECASE,
)

WILDCARD_PATTERN = re.compile(r"(?:fuck|gloryhole|porn|wank|xxx)", re.IGNORECASE)

KEYWORDS_SUBSTRING = [
    r"色情",
    r"情色",
    r"成人视频",
    r"成人影片",
    r"成人片",
    r"18禁",
    r"女优",
    r"女優",
    r"无码",
    r"無碼",
    r"无修正",
    r"無修正",
    r"内射",
    r"中出",
    r"中出し",
    r"口交",
    r"肛交",
    r"自慰",
    r"潮喷",
    r"潮吹",
    r"潮吹き",
    r"肉便器",
    r"肉棒",
    r"假阳具",
    r"淫",
    r"エッチ",
    r"えっち",
    r"濡れ",
    r"催眠",
    r"調教",
    r"陵辱",
    r"搾乳",
    r"放尿",
    r"寝取",
    r"NTR",
    r"パイズリ",
    r"手コキ",
    r"足コキ",
    r"アナル",
    r"精液",
    r"ザーメン",
    r"顔射",
    r"売春",
    r"デリヘル",
    r"パパ活",
    r"援交",
    r"隠撮",
    r"着エロ",
    r"风骚",
    r"骚货",
    r"约炮",
    r"约啪",
    r"嫖娼",
    r"乱伦",
    r"迷奸",
    r"强奸",
    r"人妻",
    r"巨乳",
    r"爆乳",
    r"美乳",
    r"痴女",
    r"熟女",
    r"露出",
    r"偷拍",
    r"盗撮",
    r"里番",
    r"工口",
    r"嫩模",
    r"外围女",
    r"楼凤",
    r"风俗",
    r"風俗",
    r"ソープ",
    r"フェラ",
    r"セックス",
    r"ヌード",
    r"エロ",
    r"官能",
    r"乱交",
    r"痴漢",
    r"おっぱい",
    r"絶頂",
    r"アクメ",
    r"オナニー",
    r"性交",
    r"性愛",
    r"做爱",
    r"性爱",
    r"车震",
    r"开房",
    r"东热",
    r"黑丝",
    r"无套",
    r"大奶",
]

SUBSTRING_PATTERN = re.compile(r"(?:" + "|".join(KEYWORDS_SUBSTRING) + r")", re.IGNORECASE)

JAV_LABEL_PREFIXES = {
    "abp", "ap", "bn", "carib", "cawd", "dvdms", "fc2", "fsdss", "fset", "gachinco", "gapl",
    "heyzo", "hodv", "homa", "ipx", "ipz", "jav", "jul", "jux", "kaad", "luxu", "mbr", "miaa",
    "miad", "mide", "midv", "mikr", "mium", "mmc", "nass", "nkd", "onsg", "pred", "rbd", "sdnm",
    "snis", "sone", "snos", "ssis", "ssni", "start", "sun", "tek", "venu", "waaa",
}

JAV_CODE_EXCLUDED_PREFIXES = {
    "aac", "ac", "amd", "audi", "avc", "bd", "benz", "bmw", "canon", "cd", "cpu", "ddp", "ddr",
    "dts", "dvd", "fhd", "ford", "fuji", "gpu", "gtx", "hdd", "hd", "hdr", "hevc", "honda",
    "hyundai", "intel", "ios", "iso", "kia", "mazda", "mlb", "mp", "mpeg", "nba", "nfl", "nhl",
    "nissan", "nvidia", "nvme", "olympus", "os", "panasonic", "pci", "psp", "rar", "rtx", "rx",
    "sata", "sdr", "sony", "ssd", "subaru", "tesla", "toyota", "ts", "ufc", "uhd", "usb", "vga",
    "vw", "web", "win", "windows", "wwe", "xbox", "zip",
}

JAV_CODE_EXCLUDED_NUMBERS = {"480", "540", "576", "720", "1080", "1440", "2160", "4320"}

JAV_CODE_PATTERN = re.compile(r"(?:^|[\W_])([a-z]{2,6})-(\d{3,6})(?:$|[\W_])", re.IGNORECASE)
JAV_COMPACT_PATTERN = re.compile(r"(?:^|[\W_])([a-z]{2,6})(\d{3,5})(?:$|[\W_])", re.IGNORECASE)


def is_mainstream_allowed(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[\W_]+", " ", text)
    return bool(MAINSTREAM_PATTERN.search(normalized))


def is_jav_code(text: str) -> bool:
    for match in JAV_CODE_PATTERN.finditer(text):
        prefix = match.group(1).lower()
        number = match.group(2)
        if prefix in JAV_CODE_EXCLUDED_PREFIXES or number in JAV_CODE_EXCLUDED_NUMBERS:
            continue
        if len(number) == 4 and number.isdigit() and 1900 <= int(number) <= 2099:
            continue
        return True
    for match in JAV_COMPACT_PATTERN.finditer(text):
        if match.group(1).lower() in JAV_LABEL_PREFIXES:
            return True
    return False


def is_keyword_nsfw(text: str) -> bool:
    if not text or is_mainstream_allowed(text):
        return False
    return bool(
        TOKEN_PATTERN.search(text)
        or WILDCARD_PATTERN.search(text)
        or SUBSTRING_PATTERN.search(text)
        or is_jav_code(text)
    )


def is_non_western(text: str) -> bool:
    return bool(NON_WESTERN_PATTERN.search(text))
