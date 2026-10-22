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
    r"loli(?:con|ta)\s+(?:nude|porn\w*|sex\w*)",
    r"lolicon",
    r"lolis?",
    r"minor\s+(?:nude|porn\w*|sex\w*)",
    r"pedo(?:phil\w*)?",
    r"preteen\s+(?:nude|porn\w*|sex\w*)",
    r"ptsc",
    r"pthc",
    r"r@ygold",
    r"reelkiddymov",
    r"shotacon",
    r"underage\s+(?:nude|porn\w*|sex\w*)",
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
    let pattern = format!(
        r"(?i)(?:(?:^|[\W_])(?:{})(?:$|[\W_])|(?:{}))",
        BANNED_KEYWORDS.join("|"),
        BANNED_SUBSTRING_KEYWORDS.join("|")
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
}
