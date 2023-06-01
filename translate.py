from langdetect import detect_langs
from googletrans import Translator

def src_lang(text):
    languages = detect_langs(text)
    likely_lang = languages[0].lang if languages else 'en'
    return likely_lang

def convert(text, lang):
    languages = detect_langs(text)
    likely_lang = languages[0].lang if languages else lang
    translator = Translator()
    translated_text = translator.translate(text, src=likely_lang, dest=lang)
    return translated_text.text

