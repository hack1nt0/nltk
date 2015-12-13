#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford Chinese Segmenter
#
# Copyright (C) 2001-2014 NLTK Project
# Author: 52nlp <52nlpcn@gmail.com>
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

import tempfile
import os
import json
from subprocess import PIPE

from nltk import compat
from nltk.internals import find_jar, config_java, java, _java_options
from nltk.tokenize.api import TokenizerI


class StanfordSegmenter(TokenizerI):
    r"""
    Interface to the Stanford Segmenter

    from nltk.tokenize.stanford_segmenter import StanfordSegmenter
    segmenter = StanfordSegmenter(path_to_jar="stanford-segmenter-3.4.1.jar", path_to_sihan_corpora_dict="./data", path_to_model="./data/pku.gz", path_to_dict="./data/dict-chris6.ser.gz", base_path='/')
    sentence = u"这是斯坦福中文分词器测试"
    segmenter.segment(sentence)
    u'\u8fd9 \u662f \u65af\u5766\u798f \u4e2d\u6587 \u5206\u8bcd\u5668 \u6d4b\u8bd5\n'
    segmenter.segment_file("test.simp.utf8")
    u'\u9762\u5bf9 \u65b0 \u4e16\u7eaa \uff0c \u4e16\u754c \u5404\u56fd ...
    """

    _JAR = 'stanford-segmenter.jar'

    def __init__(self, path_to_jar=None,
            path_to_sihan_corpora_dict=None,
            path_to_model=None, path_to_dict=None,
            encoding='UTF-8', options=None,
            verbose=False, java_options='-Xmx2g', base_path='', appendix_jar_path='', separator=' '):
        self._stanford_jar = find_jar(
            self._JAR, os.path.join(base_path, path_to_jar),
            env_vars=('STANFORD_SEGMENTER',),
            searchpath=(),
            verbose=verbose
        )

        for appendix_jar in os.listdir(appendix_jar_path):
            if not appendix_jar.endswith('.jar') or appendix_jar == path_to_jar: continue
            self._stanford_jar += ':' + os.path.join(appendix_jar_path, appendix_jar)

        self._sihan_corpora_dict = os.path.join(base_path, path_to_sihan_corpora_dict)
        self._model = os.path.join(base_path, path_to_model)
        self._dict = os.path.join(base_path, path_to_dict)

        self._encoding = encoding
        self.java_options = java_options
        options = {} if options is None else options
        self._options_cmd = ','.join('{0}={1}'.format(key, json.dumps(val)) for key, val in options.items())
        self._SEPARATOR = separator

    def segment_file(self, input_file_path):
        """
        """
        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-sighanCorporaDict', self._sihan_corpora_dict,
            '-textFile', input_file_path,
            '-sighanPostProcessing', 'true',
            '-keepAllWhitespaces', 'false',
            '-loadClassifier', self._model,
            '-serDictionary', self._dict
        ]

        stdout = self._execute(cmd)

        return stdout

    def segment(self, sentence):
        result, raw_result = self.segment_sents([sentence])
        if result and result[0] and isinstance(result[0], list) and len(result) == 1:
            result = sum(result, [])
        return result, raw_result

    def segment_sents(self, sentences):
        """
        """
        encoding = self._encoding
        # Create a temporary input file
        _input_fh, self._input_file_path = tempfile.mkstemp(text=True)

        # Write the actural sentences to the temporary input file
        _input_fh = os.fdopen(_input_fh, 'wb')
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, compat.text_type) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()

        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-sighanCorporaDict', self._sihan_corpora_dict,
            '-textFile', self._input_file_path,
            '-sighanPostProcessing', 'true',
            '-keepAllWhitespaces', 'false',
            '-loadClassifier', self._model,
            '-serDictionary', self._dict
        ]

        stdout = self._execute(cmd)

        # Delete the temporary file
        os.unlink(self._input_file_path)


        return self.parse_output(stdout, sentences), stdout

    def parse_output(self, text, sentences = None):
        # Output the tagged sentences
        segmented_sentences = []
        for segmented_sentence in text.strip().split("\n"):
            tokens = segmented_sentence.strip().split(self._SEPARATOR)
            segmented_sentences.append(tokens)
        return segmented_sentences

    def _execute(self, cmd, verbose=False):
        encoding = self._encoding
        cmd.extend(['-inputEncoding', encoding])
        _options_cmd = self._options_cmd
        if _options_cmd:
            cmd.extend(['-options', self._options_cmd])

        default_options = ' '.join(_java_options)

        # Configure java.
        config_java(options=self.java_options, verbose=verbose)

        stdout, _stderr = java(cmd,classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE)
        stdout = stdout.decode(encoding)

        # Return java configurations to their default values.
        config_java(options=default_options, verbose=False)

        return stdout
