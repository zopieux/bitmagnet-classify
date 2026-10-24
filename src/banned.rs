use regex::Regex;
use std::sync::LazyLock;

pub const BANNED_KEYWORDS: &[&str] = &[
    r"babyheart",
    r"boylover",
    r"child\s+(?:porn\w*|prostitut\w*|sex\w*)",
    r"child\s+sexual\s+abuse",
    r"childlover",
    r"csam",
    r"girllover",
    r"jailbait",
    r"kidd(?:ie|y)\s+porn\w*",
    r"loli(?:con)\s+(?:nude|porn\w*|sex\w*)",
    r"lolicon",
    r"loli\s+(?:nude|porn\w*|sex\w*)",
    r"loli(?:s|ta|tas)?\s+(?:image|images|set|sets|gallery|pack|packs|collection|model|models|dvd|photo|photos)",
    r"shotacon",
    r"minor\s+(?:nude|porn\w*|sex\w*)",
    r"pedo(?:phil\w*)?",
    r"preteens?",
    r"russkaya[\s_]*lolita",
    r"ptsc",
    r"pthc",
    r"r@ygold",
    r"reel\s*kiddymov",
    r"underage\s+(?:nude|porn\w*|sex\w*)",
    r"nymphet\w*",
    r"siberian\s*mouse",
    r"vladmodels?",
    r"shymodels?",
    r"candydoll",
    r"candy\s*girls?",
    r"babyshi(?:vid)?",
    r"hussyfan",
    r"ls\s*-?\s*(?:magazine|land|bar|stars|model|models|studio)",
    r"teen\s*(?:model|models|lolita|preteen|pre-?teen)",
    r"sandra\s*teen",
    r"karpups",
    r"little\s*cuties",
    r"darkcollection",
    r"reel\s*kiddymov",
    r"jailbat\w*",
    r"childp\w*",
    r"childporn\w*",
    r"cpack",
    r"nn\s*girls",
    r"gaylolitaboy",
    r"lolitaboy",
    r"lolifuck\w*",
    r"loliporn\w*",
    r"lolimania",
    r"lolihouse",
    r"lolitabay",
    r"lolita4u",
    r"youngfriends",
    r"youngest",
    r"pornokid",
    r"pornochild",
    r"pornchild",
    r"kinderfick\w*",
    r"kinderficken",
    r"ped[o0](?:barn|family|kid|mom|mum|mutter|dad|father|uncle|woman|child|lover|porn\w*|sex\w*|phil\w*)?",
    r"p3do\w*",
    r"ped0\w*",
    r"p[ée]do(?:phil\w*|porn\w*|fil\w*)",
    r"home\s*lolita",
    r"pteenlover",
    r"p[ée]dophil\w*",
];

pub const BANNED_SUBSTRING_KEYWORDS: &[&str] = &[
    r"儿童色情",
    r"幼女",
    r"萝莉",
    r"正太",
    r"小学生",
    r"中学生",
    r"児童ポルノ",
    r"ロリ",
    r"ショタ",
    r"援交",
    r"детск\w*\s+порн\w*",
    r"педофил\w*",
    r"несовершеннолетн\w*",
    r"лоли\w*",
    r"kinderporno\w*",
    r"p[äa]dophil\w*",
    r"ped[oó]fil\w*",
    r"p[eé]dophil\w*",
    r"pornograf\w*\s+infantil",
    r"pedopornograf\w*",
    r"pornografia\s+dzieci\w*",
];

static BANNED_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let extra = vec![
        // ages 0-17 in various forms
        r"(?:0?[1-9]|1[0-7])\s*(?:yo|y/o|y\.o\.?|yrs?|y|years?\s*old|year\s*old|ans?|anos?|años|anni|jahre|jährig\w*)",
        // loli(ta) with age
        r"loli(?:ta|tas)?[\W_]*(?:0?[1-9]|1[0-7])\b",
        r"(?:0?[1-9]|1[0-7])\s*(?:-?\s*)?(?:year|yr|y\.?o\.?)\s*(?:old|oldest)?[\W_]+loli(?:ta|tas)?",
        // loli(ta) with child/sexual context
        r"loli(?:ta|tas)?[\W_]*(?:child|kid|baby|toddler|infant|preteen|pre-?teen|underage|minor|schoolgirl|school\s*girl|incest|rape|diaper|nude|naked|porn\w*|sex\w*|fuck\w*|whore|slut|model|magazine|young\w*|tiny\w*|little\w*|ls\b|hebe|nymph\w*)",
        r"(?:child|kid|baby|toddler|infant|preteen|pre-?teen|underage|minor|young\w*|little\w*|tiny\w*|hebe|nymph\w*).{0,40}(?:loli(?:ta|tas)?|sex\w*|porn\w*|fuck\w*|nude|naked|whore|slut|rape|incest|xxx|ped\w*)",
        r"(?:sex\w*|porn\w*|fuck\w*|nude|naked|whore|slut|rape|incest|xxx).{0,40}(?:loli(?:ta|tas)?|child|kid|baby|toddler|infant|preteen|pre-?teen|underage|minor|young\w*|little\w*|tiny\w*|hebe|nymph\w*)",
        // explicit age ranges
        r"(?:\b(?:6|7|8|9|10|11|12|13|14|15|16|17)\s*(?:y|yr|yrs|yo|year|years)\b)",
    ];
    let pattern = format!(
        r"(?i)(?:(?:^|[\W_])(?:{})(?:$|[\W_])|(?:{}))|(?:{})",
        BANNED_KEYWORDS.join("|"),
        BANNED_SUBSTRING_KEYWORDS.join("|"),
        extra.join("|")
    );
    Regex::new(&pattern).expect("Valid banned pattern")
});

pub fn is_banned(text: &str) -> bool {
    !text.is_empty() && BANNED_PATTERN.is_match(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banned_keywords() {
        assert!(is_banned("pthc collection"));
        assert!(is_banned("PTHC-2024"));
        assert!(is_banned("ptsc videoclip"));
        assert!(is_banned("pedo party"));
        assert!(is_banned("Pedophile material"));
        assert!(is_banned("child porn"));
        assert!(is_banned("child prostitute"));
        assert!(is_banned("preteen sex"));
        assert!(is_banned("underage nude"));
        assert!(is_banned("r@ygold archive"));
        assert!(is_banned("babyheart"));
        assert!(is_banned("Loli image set"));
        assert!(is_banned("这是儿童色情视频"));
        assert!(is_banned("幼女视频"));
        assert!(is_banned("ロリ image"));
        assert!(is_banned("педофил"));
        assert!(is_banned("детское порно"));
        assert!(is_banned("kinderporno"));
        assert!(is_banned("pornografía infantil"));
    }

    #[test]
    fn test_mainstream_not_banned() {
        assert!(!is_banned(""));
        assert!(!is_banned("The Children of Dune (2003)"));
        assert!(!is_banned("Childhood's End - Arthur C. Clarke"));
        assert!(!is_banned("Lolita (1962)"));
        assert!(!is_banned("Underage Drinking Documentary"));
        assert!(!is_banned("pedometer workout"));
        assert!(!is_banned("The Child"));
    }

    #[test]
    fn test_lolita_age_variants() {
        assert!(is_banned("incesto lolita 5y CBaby Pedoman.mpg"));
        assert!(is_banned("ls-magazine Japanese lolita 09yo Blue Orchid lolifuck 12yo+9yo.avi"));
        assert!(is_banned("Home Lolita - 16.rar"));
        assert!(is_banned("Japanese lolita 6 year old whore! .mpg"));
        assert!(is_banned("NoLimitsFun mylola manami 11yo Girlchild Romanian preteen lolita novinha.wmv"));
        assert!(is_banned("10yrs child cjhildporn Russian lolita.mpeg"));
        assert!(is_banned("darkcollection hebe kid bd girls black lolita.mpg"));
        assert!(is_banned("MashaWorld Young lolita compilation Tara 12yo 13yrs.mp4"));
        assert!(is_banned("10yrs nenas de argentina lsm  speedo Dad and Daughter lolipop 1yo (lolita anal).mpg"));
        assert!(is_banned("russkaya_lolita.avi"));
        assert!(is_banned("Pteenlover (lolita anal) screaming baby in diapers.wmv"));
        assert!(is_banned("little girl xxx"));
    }
}
