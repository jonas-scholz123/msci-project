def get_id2tag(corpus, detail_level=0):

    if corpus == "mrda":
        if detail_level == 0:
            return {0: "Q", 1: "D", 2: "F", 3: "S", 4: "B"}

        elif detail_level == 1:
            return {
                0: "h",
                1: "qw",
                2: "fh",
                3: "fg",
                4: "qr",
                5: "qo",
                6: "qh",
                7: "qy",
                8: "qrr",
                9: "s",
                10: "b",
                11: "%",
            }

        elif detail_level == 2:
            return {
                0: "s",
                1: "b",
                2: "fh",
                3: "bk",
                4: "aa",
                5: "df",
                6: "e",
                7: "%",
                8: "rt",
                9: "fg",
                10: "cs",
                11: "ba",
                12: "bu",
                13: "d",
                14: "na",
                15: "qw",
                16: "ar",
                17: "2",
                18: "no",
                19: "h",
                20: "co",
                21: "qy",
                22: "nd",
                23: "j",
                24: "bd",
                25: "cc",
                26: "ng",
                27: "am",
                28: "qrr",
                29: "fe",
                30: "m",
                31: "fa",
                32: "t",
                33: "br",
                34: "aap",
                35: "qh",
                36: "tc",
                37: "r",
                38: "t1",
                39: "t3",
                40: "bh",
                41: "bsc",
                42: "arp",
                43: "bs",
                44: "f",
                45: "qr",
                46: "ft",
                47: "g",
                48: "qo",
                49: "bc",
                50: "by",
                51: "fw",
            }
    elif corpus == "swda":
        return {
            0: "sd",
            1: "b",
            2: "sv",
            3: "%",
            4: "aa",
            5: "ba",
            6: "qy",
            7: "ny",
            8: "fc",
            9: "qw",
            10: "nn",
            11: "bk",
            12: "h",
            13: "qy^d",
            14: "bh",
            15: "^q",
            16: "bf",
            17: 'fo_o_fw_"_by_bc',
            18: "na",
            19: "ad",
            20: "^2",
            21: "b^m",
            22: "qo",
            23: "qh",
            24: "^h",
            25: "ar",
            26: "ng",
            27: "br",
            28: "no",
            29: "fp",
            30: "qrr",
            31: "arp_nd",
            32: "t3",
            33: "oo_co_cc",
            34: "aap_am",
            35: "t1",
            36: "bd",
            37: "^g",
            38: "qw^d",
            39: "fa",
            40: "ft",
        }


def get_tag2full_label(corpus, detail_level):

    if corpus == "mrda":
        if detail_level == 0:
            return {
                "S": "Statement",
                "B": "Backchannel",
                "D": "Disruption",
                "F": "FloorGrabber",
                "Q": "Question",
            }

        if detail_level > 0:  # can be reused for medium tags
            return {
                "s": "Statement",
                "b": "Continuer",
                "fh": "FloorHolder",
                "bk": "Acknowledge-answer",
                "aa": "Accept",
                "df": "Defending/Explanation",
                "e": "Expansionsofy/nAnswers",
                "%": "Interrupted/Abandoned/Uninterpretable",
                "rt": "RisingTone",
                "fg": "FloorGrabber",
                "cs": "Offer",
                "ba": "Assessment/Appreciation",
                "bu": "UnderstandingCheck",
                "d": "Declarative-Question",
                "na": "AffirmativeNon-yesAnswers",
                "qw": "Wh-Question",
                "ar": "Reject",
                "2": "CollaborativeCompletion",
                "no": "2OtherAnswers",
                "h": "HoldBeforeAnswer/Agreement",
                "co": "Action-directive",
                "qy": "Yes-No-question",
                "nd": "DispreferredAnswers",
                "j": "HumorousMaterial",
                "bd": "Downplayer",
                "cc": "Commit",
                "ng": "NegativeNon-noAnswers",
                "am": "Maybe",
                "qrr": "Or-Clause",
                "fe": "Exclamation",
                "m": "MimicOther",
                "fa": "Apology",
                "t": "About-task",
                "br": "Signal-non-understanding",
                "aap": "Accept-part",
                "qh": "Rhetorical-Question",
                "tc": "TopicChange",
                "r": "Repeat",
                "t1": "Self-talk",
                "t3": "3rd-party-talk",
                "bh": "Rhetorical-questionContinue",
                "bsc": "Reject-part",
                "arp": "MisspeakSelf-Correction",
                "bs": "Reformulate/Summarize",
                "f": 'FollowMe"',
                "qr": "Or-Question",
                "ft": "Thanking",
                "g": "Tag-Question",
                "qo": "Open-Question",
                "bc": "Correct-misspeaking",
                "by": "Sympathy",
                "fw": "Welcome",
            }

    if corpus == "swda":
        return {
            "sd": "Statement-non-opinion",
            "b": "Acknowledge (Backchannel)",
            "sv": "Statement-opinion",
            "aa": "Agree/Accept",
            "% -": "Abandoned or Turn-Exit",
            "ba": "Appreciation",
            "qy": "Yes-No-Question",
            "x": "Non-verbal",
            "ny": "Yes answers",
            "fc": "Conventional-closing",
            "%": "Uninterpretable",
            "qw": "Wh-Question",
            "nn": "No answers",
            "bk": "Response Acknowledgement",
            "h": "Hedge",
            "qy^d": "Declarative Yes-No-Question",
            'fo_o_fw_"_by_bc': "Other",
            "o": "Other",
            "fo": "Other",
            "bc": "Other",
            "by": "Other",
            "fw": "Other",
            "bh": "Backchannel in question form",
            "^q": "Quotation",
            "bf": "Summarize/reformulate",
            "ny^e": "Affirmative non-yes answers",
            "na": "Affirmative non-yes answers",
            "ad": "Action-directive",
            "^2": "Collaborative Completion",
            "b^m": "Repeat-phrase",
            "qo": "Open-Question",
            "qh": "Rhetorical-Questions",
            "^h": "Hold before answer/agreement",
            "ar": "Reject",
            "nn^e": "Negative non-no answers",
            "ng": "Negative non-no answers",
            "br": "Signal-non-understanding",
            "no": "Other answers",
            "fp": "Conventional-opening",
            "qrr": "Or-Clause",
            "nd": "Dispreferred answers",
            "arp": "Dispreferred answers",
            "t3": "3rd-party-talk",
            "oo,cc,co": "Offers, Options Commits",
            "oo_co_cc": "Offers, Options Commits",
            "t1": "Self-talk",
            "bd": "Downplayer",
            "aap/am": "Maybe/Accept-part",
            "aap_am": "Maybe/Accept-part",
            "^g": "Tag-Question",
            "qw^d": "Declarative Wh-Question",
            "fa": "Apology",
            "ft": "Thanking",
        }
