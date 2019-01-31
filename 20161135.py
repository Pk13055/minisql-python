#!/usr/bin/env python3
"""
:author: Pratik Kamble
:brief: This script acts like a mini sql engine and is used to create and
        modify tables, add records, and modify existing ones. It throws the
        appropriate arguments for missing values or incorrect queries.
:date: 30th January, 2019
"""
from itertools import chain
import json

import sqlparse


METAFILE = "metadata.txt"
printj = lambda val: print(json.dumps(val, indent=4)) if type(val) == dict else None


def process_meta():
    """Process the METAFILE and extract
    information for parsing and display

    :param: None
    :return: tuple
        - tables -> list: List of table names (name of csvs)
        - idt -> dict: Inverse attribute mapping for quick lookup
        - meta -> dict: table to attribute mapping
    """
    meta = [_.strip(' ') for _ in open(METAFILE).read().splitlines()]
    idxs = [idx for idx, val in enumerate(meta) if "<" in val]
    tables = [meta[i + 1:j] for i, j in zip(idxs[::2], idxs[1::2])]
    meta = {table[0]:table[1:] for table in tables}
    attr = list(meta.values())
    cols = set(chain(*attr))
    idt = { col: [idx for idx, ls in enumerate(attr) if col in ls] for col in cols}
    tables = list(meta.keys())
    return tables, idt, meta


def main():
    """parses the query and applies changes according to the documentation

    :params: None
    :return: None
    """
    tables, idt, meta = process_meta()
    printj(idt)
    printj(meta)

if __name__ == "__main__":
    main()
