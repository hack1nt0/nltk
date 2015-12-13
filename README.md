# Natural Language Toolkit (NLTK) with Stanford NLP package

Added Stanford's multilingual segmenter to NLTK, and tuned the original stanford api.

## Usage

``` python
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tag.stanford_tagger import StanfordNERTagger
from nltk.tag.stanford_tagger import StanfordPOSTagger

segmenter = StanfordSegmenter(base_path='/Users/DY/Downloads/stanford-segmenter', path_to_jar='stanford-segmenter.jar', path_to_sihan_corpora_dict='data', path_to_model='data/pku.gz', path_to_dict='data/dict-chris6.ser.gz', appendix_jar_path='/Users/DY/Downloads/stanford-segmenter/log4j')
strs = [u'我在做中文分词.', u'你好!']
result, raw_result = segmenter.segment_sents(strs)
print(result)
print(raw_result)

tagger = StanfordPOSTagger(path_to_model='models/chinese-distsim.tagger', path_to_jar='stanford-postagger.jar', base_path='/Users/DY/Downloads/stanford-postagger-full', appendix_jar_path='/Users/DY/Downloads/stanford-segmenter/log4j')
result, raw_result = tagger.tag_sents([[u'我', u'在', u'做', u'中文', u'分词', u'.'], [u'你', u'好', u'!']])
print(raw_result)
print(result)

ner = StanfordNERTagger(path_to_jar='stanford-ner.jar', path_to_model='classifiers/chinese.misc.distsim.crf.ser.gz', base_path='/Users/DY/Downloads/stanford-ner', appendix_jar_path='/Users/DY/Downloads/stanford-segmenter/log4j')
result, raw_result = ner.tag_sents([[u'我', u'在', u'做', u'中文', u'分词'], [u'中国', u'中央电视台']])
print raw_result
print result
```
