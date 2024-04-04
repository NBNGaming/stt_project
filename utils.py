import json
import re
from difflib import SequenceMatcher

WHISPER_CODE_TO_NAME = {
    'en': 'english',
    'zh': 'mandarin',
    'de': 'german',
    'es': 'castilian',
    'ru': 'russian',
    'ko': 'korean',
    'fr': 'french',
    'ja': 'japanese',
    'pt': 'portuguese',
    'tr': 'turkish',
    'pl': 'polish',
    'ca': 'valencian',
    'nl': 'flemish',
    'ar': 'arabic',
    'sv': 'swedish',
    'it': 'italian',
    'id': 'indonesian',
    'hi': 'hindi',
    'fi': 'finnish',
    'vi': 'vietnamese',
    'he': 'hebrew',
    'uk': 'ukrainian',
    'el': 'greek',
    'ms': 'malay',
    'cs': 'czech',
    'ro': 'moldovan',
    'da': 'danish',
    'hu': 'hungarian',
    'ta': 'tamil',
    'no': 'norwegian',
    'th': 'thai',
    'ur': 'urdu',
    'hr': 'croatian',
    'bg': 'bulgarian',
    'lt': 'lithuanian',
    'la': 'latin',
    'mi': 'maori',
    'ml': 'malayalam',
    'cy': 'welsh',
    'sk': 'slovak',
    'te': 'telugu',
    'fa': 'persian',
    'lv': 'latvian',
    'bn': 'bengali',
    'sr': 'serbian',
    'az': 'azerbaijani',
    'sl': 'slovenian',
    'kn': 'kannada',
    'et': 'estonian',
    'mk': 'macedonian',
    'br': 'breton',
    'eu': 'basque',
    'is': 'icelandic',
    'hy': 'armenian',
    'ne': 'nepali',
    'mn': 'mongolian',
    'bs': 'bosnian',
    'kk': 'kazakh',
    'sq': 'albanian',
    'sw': 'swahili',
    'gl': 'galician',
    'mr': 'marathi',
    'pa': 'panjabi',
    'si': 'sinhalese',
    'km': 'khmer',
    'sn': 'shona',
    'yo': 'yoruba',
    'so': 'somali',
    'af': 'afrikaans',
    'oc': 'occitan',
    'ka': 'georgian',
    'be': 'belarusian',
    'tg': 'tajik',
    'sd': 'sindhi',
    'gu': 'gujarati',
    'am': 'amharic',
    'yi': 'yiddish',
    'lo': 'lao',
    'uz': 'uzbek',
    'fo': 'faroese',
    'ht': 'haitian',
    'ps': 'pushto',
    'tk': 'turkmen',
    'nn': 'nynorsk',
    'mt': 'maltese',
    'sa': 'sanskrit',
    'lb': 'letzeburgesch',
    'my': 'burmese',
    'bo': 'tibetan',
    'tl': 'tagalog',
    'mg': 'malagasy',
    'as': 'assamese',
    'tt': 'tatar',
    'haw': 'hawaiian',
    'ln': 'lingala',
    'ha': 'hausa',
    'ba': 'bashkir',
    'jw': 'javanese',
    'su': 'sundanese',
    'yue': 'cantonese',
}

with open('base_hallucination_filter.json', 'r', encoding='utf-8') as f:
    BASE_FILTER = json.load(f)


def normalize_strings(str_list):
    ptn = '["\',.?!]+$'
    result = []
    for s in str_list:
        result.append(re.sub(ptn, '', s.strip()).lower())
    return result


def remove_hallucinations(segments, lang):
    lang_name = lang if len(lang) > 3 else WHISPER_CODE_TO_NAME.get(lang, '')
    str_to_find = BASE_FILTER.get(lang_name, [])

    all_segments_text = normalize_strings(segments)
    str_to_find = normalize_strings(str_to_find)

    for i, full_segment in reversed(list(enumerate(all_segments_text))):
        if any(SequenceMatcher(None, to_find, full_segment).ratio() >= 0.8 for to_find in str_to_find):
            segments.pop(i)
