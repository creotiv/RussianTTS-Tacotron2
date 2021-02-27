""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
import os
from unidecode import unidecode
from .numb import normalize_numbers
from transliterate import translit
from .emphasizer import Emphasizer

if __file__:
  rdc = Emphasizer(os.path.join(os.path.dirname(__file__),'ru_emphasize.dict'))

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

_sep = re.compile(r'[\?\,\.!\-\—]+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

def remove_sep(text):
  return re.sub(_sep, '', text)

def convert_to_ascii(text):
  return unidecode(text)

def convert_to_ascii_ua(text):
  return translit(text, 'uk', reversed=True)

def clean_end(txt):
    if txt[-1] == ',':
        txt = txt.rstrip(',')+'.'
    elif txt[-1] not in ['!','?','.']:
        txt += '.'
    return txt

def accent(txt):
  return rdc.text_to_text(txt)

def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def transliteration_ua_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii_ua(text)
  text = lowercase(text)
  # text = remove_sep(text)
  text = collapse_whitespace(text)
  text = clean_end(text)
  return text

def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = lowercase(text)
  text = convert_to_ascii(text)
  text = collapse_whitespace(text)
  text = clean_end(text)
  return text

def transliteration_cleaners_with_stress(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = lowercase(text)
  text = rdc.add_stress(text)
  text = convert_to_ascii(text)
  text = collapse_whitespace(text)
  text = clean_end(text)
  text = "S"+text+"E"
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

