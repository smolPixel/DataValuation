import unittest
import Classifier.jiant.utils.retokenize as retokenize


class TestRetokenize(unittest.TestCase):
    def setUp(self):
        self.text = [
            "Members of the House clapped their hands",
            "I look at Sarah's dog. It was cute.!",
            "Mr. Immelt chose to focus on the incomprehensibility of accounting rules.",
            "What?",
        ]
        self.token_index_src = [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0],
        ]
        self.span_index_src = [
            [(0, 4), (5, 7)],
            [(0, 1), (3, 5)],
            [(0, 2), (6, 11), (6, 8), (7, 11)],
            [(0, 1)],
        ]

    def test_moses(self):
        self.tokens = [
            ["Members", "of", "the", "House", "clapped", "their", "hands"],
            ["I", "look", "at", "Sarah", "&apos;s", "dog", ".", "It", "was", "cute", ".", "!"],
            [
                "Mr.",
                "Immelt",
                "chose",
                "to",
                "focus",
                "on",
                "the",
                "incomprehensibility",
                "of",
                "accounting",
                "rules",
                ".",
            ],
            ["What", "?"],
        ]
        self.token_index_tgt = [
            [[0], [1], [2], [3], [4], [5], [6]],
            [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]],
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10, 11]],
            [[0, 1]],
        ]
        self.span_index_tgt = [
            [(0, 4), (5, 7)],
            [(0, 1), (3, 7)],
            [(0, 2), (6, 12), (6, 8), (7, 12)],
            [(0, 2)],
        ]

        aligner_fn = retokenize.get_aligner_fn("transfo-xl-wt103")
        token_aligners, tokens = zip(*(aligner_fn(sent) for sent in self.text))
        token_aligners, tokens = list(token_aligners), list(tokens)
        token_index_tgt = [
            [token_aligner.project_tokens(idxs).tolist() for idxs in token_idxs]
            for token_aligner, token_idxs in zip(token_aligners, self.token_index_src)
        ]
        span_index_tgt = [
            [token_aligner.project_span(start, end) for (start, end) in span_idxs]
            for token_aligner, span_idxs in zip(token_aligners, self.span_index_src)
        ]
        assert self.tokens == tokens
        assert self.token_index_tgt == token_index_tgt
        assert self.span_index_tgt == span_index_tgt

    def test_wpm(self):
        self.tokens = [
            ["Members", "of", "the", "House", "clapped", "their", "hands"],
            ["I", "look", "at", "Sarah", "'", "s", "dog", ".", "It", "was", "cute", ".", "!"],
            [
                "Mr",
                ".",
                "I",
                "##mme",
                "##lt",
                "chose",
                "to",
                "focus",
                "on",
                "the",
                "in",
                "##com",
                "##p",
                "##re",
                "##hen",
                "##si",
                "##bility",
                "of",
                "accounting",
                "rules",
                ".",
            ],
            ["What", "?"],
        ]
        self.token_index_tgt = [
            [[0], [1], [2], [3], [4], [5], [6]],
            [[0], [1], [2], [3, 4, 5], [6, 7], [8], [9], [10, 11, 12]],
            [
                [0, 1],
                [2, 3, 4],
                [5],
                [6],
                [7],
                [8],
                [9],
                [10, 11, 12, 13, 14, 15, 16],
                [17],
                [18],
                [19, 20],
            ],
            [[0, 1]],
        ]
        self.span_index_tgt = [
            [(0, 4), (5, 7)],
            [(0, 1), (3, 8)],
            [(0, 5), (9, 21), (9, 17), (10, 21)],
            [(0, 2)],
        ]

        aligner_fn = retokenize.get_aligner_fn("bert-base-cased")
        token_aligners, tokens = zip(*(aligner_fn(sent) for sent in self.text))
        token_aligners, tokens = list(token_aligners), list(tokens)
        token_index_tgt = [
            [token_aligner.project_tokens(idxs).tolist() for idxs in token_idxs]
            for token_aligner, token_idxs in zip(token_aligners, self.token_index_src)
        ]
        span_index_tgt = [
            [token_aligner.project_span(start, end) for (start, end) in span_idxs]
            for token_aligner, span_idxs in zip(token_aligners, self.span_index_src)
        ]
        assert self.tokens == tokens
        assert self.token_index_tgt == token_index_tgt
        assert self.span_index_tgt == span_index_tgt

    def test_bpe(self):
        self.tokens = [
            [
                "members</w>",
                "of</w>",
                "the</w>",
                "house</w>",
                "clapped</w>",
                "their</w>",
                "hands</w>",
            ],
            [
                "i</w>",
                "look</w>",
                "at</w>",
                "sarah</w>",
                "'s</w>",
                "dog</w>",
                ".</w>",
                "it</w>",
                "was</w>",
                "cute</w>",
                ".</w>",
                "!</w>",
            ],
            [
                "mr.</w>",
                "im",
                "melt</w>",
                "chose</w>",
                "to</w>",
                "focus</w>",
                "on</w>",
                "the</w>",
                "in",
                "comprehen",
                "si",
                "bility</w>",
                "of</w>",
                "accounting</w>",
                "rules</w>",
                ".</w>",
            ],
            ["what</w>", "?</w>"],
        ]
        self.token_index_tgt = [
            [[0], [1], [2], [3], [4], [5], [6]],
            [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]],
            [[0], [1, 2], [3], [4], [5], [6], [7], [8, 9, 10, 11], [12], [13], [14, 15]],
            [[0, 1]],
        ]
        self.span_index_tgt = [
            [(0, 4), (5, 7)],
            [(0, 1), (3, 7)],
            [(0, 3), (7, 16), (7, 12), (8, 16)],
            [(0, 2)],
        ]

        aligner_fn = retokenize.get_aligner_fn("openai-gpt")
        token_aligners, tokens = zip(*(aligner_fn(sent) for sent in self.text))
        token_aligners, tokens = list(token_aligners), list(tokens)
        token_index_tgt = [
            [token_aligner.project_tokens(idxs).tolist() for idxs in token_idxs]
            for token_aligner, token_idxs in zip(token_aligners, self.token_index_src)
        ]
        span_index_tgt = [
            [token_aligner.project_span(start, end) for (start, end) in span_idxs]
            for token_aligner, span_idxs in zip(token_aligners, self.span_index_src)
        ]
        assert self.tokens == tokens
        assert self.token_index_tgt == token_index_tgt
        assert self.span_index_tgt == span_index_tgt

    def test_sentencepiece(self):
        self.tokens = [
            ["▁Members", "▁of", "▁the", "▁House", "▁clapped", "▁their", "▁hands"],
            [
                "▁I",
                "▁look",
                "▁at",
                "▁Sarah",
                "'",
                "s",
                "▁dog",
                ".",
                "▁It",
                "▁was",
                "▁cute",
                ".",
                "!",
            ],
            [
                "▁Mr",
                ".",
                "▁I",
                "m",
                "mel",
                "t",
                "▁chose",
                "▁to",
                "▁focus",
                "▁on",
                "▁the",
                "▁in",
                "comp",
                "re",
                "hen",
                "s",
                "ibility",
                "▁of",
                "▁accounting",
                "▁rules",
                ".",
            ],
            ["▁What", "?"],
        ]
        self.token_index_tgt = [
            [[0], [1], [2], [3], [4], [5], [6]],
            [[0], [1], [2], [3, 4, 5], [6, 7], [8], [9], [10, 11, 12]],
            [
                [0, 1],
                [2, 3, 4, 5],
                [6],
                [7],
                [8],
                [9],
                [10],
                [11, 12, 13, 14, 15, 16],
                [17],
                [18],
                [19, 20],
            ],
            [[0, 1]],
        ]
        self.span_index_tgt = [
            [(0, 4), (5, 7)],
            [(0, 1), (3, 8)],
            [(0, 6), (10, 21), (10, 17), (11, 21)],
            [(0, 2)],
        ]

        aligner_fn = retokenize.get_aligner_fn("xlnet-base-cased")
        token_aligners, tokens = zip(*(aligner_fn(sent) for sent in self.text))
        token_aligners, tokens = list(token_aligners), list(tokens)
        token_index_tgt = [
            [token_aligner.project_tokens(idxs).tolist() for idxs in token_idxs]
            for token_aligner, token_idxs in zip(token_aligners, self.token_index_src)
        ]
        span_index_tgt = [
            [token_aligner.project_span(start, end) for (start, end) in span_idxs]
            for token_aligner, span_idxs in zip(token_aligners, self.span_index_src)
        ]
        assert self.tokens == tokens
        assert self.token_index_tgt == token_index_tgt
        assert self.span_index_tgt == span_index_tgt

    def test_bytebpe(self):
        self.tokens = [
            ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"],
            ["I", "Ġlook", "Ġat", "ĠSarah", "'s", "Ġdog", ".", "ĠIt", "Ġwas", "Ġcute", ".", "!"],
            [
                "Mr",
                ".",
                "ĠImm",
                "elt",
                "Ġchose",
                "Ġto",
                "Ġfocus",
                "Ġon",
                "Ġthe",
                "Ġincomp",
                "rehens",
                "ibility",
                "Ġof",
                "Ġaccounting",
                "Ġrules",
                ".",
            ],
            ["What", "?"],
        ]
        self.token_index_tgt = [
            [[0], [1], [2], [3], [4, 5], [6], [7]],
            [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]],
            [[0, 1], [2, 3], [4], [5], [6], [7], [8], [9, 10, 11], [12], [13], [14, 15]],
            [[0, 1]],
        ]
        self.span_index_tgt = [
            [(0, 4), (6, 8)],
            [(0, 1), (3, 7)],
            [(0, 4), (8, 16), (8, 12), (9, 16)],
            [(0, 2)],
        ]

        aligner_fn = retokenize.get_aligner_fn("roberta-base")
        token_aligners, tokens = zip(*(aligner_fn(sent) for sent in self.text))
        token_aligners, tokens = list(token_aligners), list(tokens)
        token_index_tgt = [
            [token_aligner.project_tokens(idxs).tolist() for idxs in token_idxs]
            for token_aligner, token_idxs in zip(token_aligners, self.token_index_src)
        ]
        span_index_tgt = [
            [token_aligner.project_span(start, end) for (start, end) in span_idxs]
            for token_aligner, span_idxs in zip(token_aligners, self.span_index_src)
        ]
        assert self.tokens == tokens
        assert self.token_index_tgt == token_index_tgt
        assert self.span_index_tgt == span_index_tgt
