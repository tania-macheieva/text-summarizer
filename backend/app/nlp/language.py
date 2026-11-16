"""Мовні ресурси: визначення мови, стоп-слова, стемер."""
from __future__ import annotations

import re
from functools import lru_cache

SUPPORTED_LANGUAGES = ("en", "uk")

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")

# Вбудований список українських стоп-слів (NLTK не містить українського списку).
UK_STOPWORDS = frozenset(
    """і й та а але або чи що щоб як коли де тому бо ж же би б ні не ані так то це цей ця ці той та те
    ті цього цієї цьому цим цих тих тієї того тим я ми ти ви він вона воно вони мене мені мною нас нам
    нами вас вам вами його її їх йому їй ним нею ними себе собі собою мій моя моє мої твій твоя твоє
    наш наша наше наші ваш ваша ваше ваші свій своя своє свої який яка яке які якого якої яких якому
    ким чим чого кого кому хто що все всі всього всіх вся весь уся усе кожен кожна кожне кожні
    є був була було були буде будуть бути буду будемо будете будь можна треба можуть може мав мала мало
    мали маєте мають мати у в во з із зі до від від до на над під при про через між за без для по
    після перед поза біля серед навколо проти крім ще вже тільки лише навіть також теж дуже більш
    менш там тут тоді тепер зараз завжди ніколи іноді саме ось оце ніж аж""".split()
)

_EN_FALLBACK_STOPWORDS = frozenset(
    """a an the and or but if while of at by for with about against between into through during before
    after above below to from up down in out on off over under again further then once here there when
    where why how all any both each few more most other some such no nor not only own same so than too
    very s t can will just don should now i me my myself we our ours you your he him his she her it its
    they them their what which who whom this that these those am is are was were be been being have has
    had having do does did doing would could also said says say""".split()
)


def detect_language(text: str) -> str:
    """Евристика за часткою кирилиці/латиниці; повертає 'uk' або 'en'."""
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    if cyr == 0 and lat == 0:
        return "en"
    return "uk" if cyr > lat else "en"


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        words = frozenset(stopwords.words("english"))
    except Exception:  # ресурс не завантажено — використовуємо вбудований список
        words = frozenset()
    return words | _EN_FALLBACK_STOPWORDS


def stopwords_for(language: str) -> frozenset[str]:
    return UK_STOPWORDS if language == "uk" else english_stopwords()


@lru_cache(maxsize=1)
def english_stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()
