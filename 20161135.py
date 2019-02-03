#!/usr/bin/env python3
"""
:author: Pratik Kamble
:brief: This script acts like a mini sql engine and is used to create and
        modify tables, add records, and modify existing ones. It throws the
        appropriate arguments for missing values or incorrect queries.
:date: 30th January, 2019
"""
import csv
from datetime import datetime as d
from itertools import chain
from sys import argv as rd
import json

import sqlparse


METAFILE = "metadata.txt"
printj = lambda val: print(json.dumps(val, indent=4)) if type(val) == dict else None


class QueryParser(object):
    """QueryParser is responsible for parsing the given query,
    given table and schema constraints. Since datatype of the
    records is only integers, no type checking is done
    """

    def __init__(self, tables, idt, meta):
        self.start_session = d.now()
        self.end_session = None

        self.tables = tables
        self.idt = idt
        self.meta = meta
        self.parse_col = lambda val: val.split('.') if '.' in val else [None, val]


        self.query = None
        self.res = None
        self.dialects = [csv.Sniffer().sniff(open(table + ".csv").read(1024)) for
            table in self.tables]
        self.history = []


    def __str__(self):
        return json.dumps(meta)

    def save_history(self, filename: str) -> bool:
        """Saves the history of the outputs to a file
        :param filename: str -> Filename to save in
        :return success: bool -> Whether saved successfully
        """
        try:
            self.end_session = d.now()
            with open(filename, 'a+') as f:
                f.write(f"{10 * '='} New Session {10 * '='}\n")
                f.write(str(self.start_session))
                f.write(f"\n{20 * '*'}\n".join(self.history))
                f.write(f"\n{str(self.end_session)}\n")
                f.write(f"{10 * '='} End Session {10 * '='}\n")
            if len(self.history) == 0:
                print("sqlwarn: Empty result history")
            self.history = []
            return True
        except Exception as e:
            print(f"sqlerror: {e}")
            return False

    def parse_function(self, fn: sqlparse.sql.Function) -> list:
        """Parses the function token to extract table and function

        :param fn_token: sqlparse.sql.Function -> the function token
        :return simple_token: list[list] -> the list of function and
        normalized column name
        """
        col_name = fn.get_parameters()[0].normalized
        return [fn.get_name(), self.parse_col(col_name)]

    def parse_fields(self, base: list, columns: list) -> list:
        """Parses the fields into a workable format

        :param base: base field params
        :param columns: valid columns
        :return fields: list -> A list of the format
            - [
                {
                    field: 'tableX.A',
                    function: 'foo' or NONE or distinct
                }, ...
            ]
        """
        tokens = []
        skip = False
        try:
            for idx, token in enumerate(base):
                simple_token = []
                if skip is True:
                    skip = False
                    continue
                if token.ttype == sqlparse.tokens.Keyword:
                    simple_token = [token.normalized, self.parse_col(base[idx + 1].normalized)]
                    skip = True
                elif isinstance(token, sqlparse.sql.Function):
                    simple_token = self.parse_function(token)
                else:
                    simple_token = self.parse_col(token.normalized)
                tokens.append(simple_token)
        except Exception as e:
            print(f"tokenerror: unsupported format | {e}")
            tokens = []
        return tokens

    def parse_where(self, base: list) -> list:
        """Parses the where section to extract join params

        :param base: list -> Base set of tokens to parse from
        This contains the raw where statement
        :return where_sec: dict -> where part of the `simplified_tokens`
        that contains any possible comparisons
        """
        where_sec = {}
        print(base)
        return where_sec

    def analyze_tokens(self, base: list) -> dict:
        """Analyse tokens and parse them into a simple list
        :param base: list[sqlparse.tokens] -> base set of sqlparse tokens
        :return simplified_tokens: list[str] -> simple token system token
            fetch the result. The result token list is going to be of the following:
                - columns to be fetched -> tableX.colY, ...
                    - Note: any function would be listed as
                    func_name(tableX.colY)
                - conditions -> tableX.colY <OP> const OR tableX.colY
                - joined by at most one AND/OR
                [['table1', 'table2', ...], [tableX.colY, func(...), ...], ]
        """

        simplified_tokens = {}
        last_token = base[-1]  # either a WHERE statement or a punctutation
        table_list = base[-2]  # tables being referenced for the columns
        fields = base[:-3]  # a list of fields and functions being used

        # check whether tables exist
        table_list = [_ for _ in table_list if _.ttype not in [
            sqlparse.tokens.Punctuation, sqlparse.tokens.Whitespace]]
        if not all([_.normalized in self.tables for _ in table_list]):
            raise Exception(f"sqlerror: Missing table referenced")

        simplified_tokens['tables'] = [_.normalized for _ in table_list]
        valid_columns = set(chain(*[self.meta[_.normalized] for _ in table_list]))

        # parse fields to retrieve
        fields = [_ for ls in fields for _ in ls if _.ttype not in
                  [sqlparse.tokens.Whitespace, sqlparse.tokens.Punctuation]]
        simplified_tokens['fields'] = self.parse_fields(fields, valid_columns)

        # parse where section if exists
        if last_token.ttype != sqlparse.tokens.Punctuation:
            simplified_tokens['where'] = self.parse_where(last_token)
        return simplified_tokens


    def compile_result(self, tokens: list) -> str:
        """Compile the results after verifying valid statement
        Handle the where condition and comparators and actual data fetching

        :param tokens: list -> simplified list of tokens to parse
        :return res: str -> Resulting formatted result depending on the query
        """
        res = ""
        printj(tokens)
        return res

    def parse(self, query: sqlparse.sql.Statement) -> str:
        """Parses a single query and returns the result.
        Optionally, the result is also stored in the history

        :param query: sqlparse statment -> The query to be parsed
        :return result: str -> the record result to be parsed
        """
        # TODO: add query parsing here (only select statements supported)
        res = ""

        # Note: self.query and res contain the result and query that
        # was last executed successfully
        if query[0].normalized == "SELECT" and query[-1].normalized[-1] == ';':
            self.query = query
            base = [_ for _ in self.query if _.normalized != ' ']
            try:
                tokens = self.analyze_tokens(base[1:])
                res = self.compile_result(tokens)
                self.query = query
            except Exception as e:
                print(f"[LOOKUP_ERROR] {e}")
        else:
            res = f"sqlerror: unsupported statement {self.query} (NOT_SELECT)"
        print(res)
        self.history.append(res)
        return res


def process_meta():
    """Process the METAFILE and extract
    information for parsing and display

    :param: None
    :return: tuple
        - tables -> list: List of table names (name of csvs)
        - idt -> dict: Inverse attribute mapping for quick lookup
        - meta -> dict: table to attribute mapping
    """
    try:
        meta = [_.strip(' ') for _ in open(METAFILE).read().splitlines()]
        idxs = [idx for idx, val in enumerate(meta) if "<" in val]
        tables = [meta[i + 1:j] for i, j in zip(idxs[::2], idxs[1::2])]
        meta = {table[0]:table[1:] for table in tables}
        attr = list(meta.values())
        cols = set(chain(*attr))
        idt = { col: [idx for idx, ls in enumerate(attr) if col in ls] for col in cols}
        tables = list(meta.keys())
    except Exception as e:
        print(f"metaerror: {e}")
        tables, idt, meta = [], {}, {}
    return tables, idt, meta


def main():
    """parses the query and applies changes according to the documentation

    :params: None
    :return: None
    """
    tables, idt, meta = process_meta()
    try:
        queries = sqlparse.parse(rd[1])
    except Exception as e:
        print(f"sqlerror: Parsing failed {e}")
        return -1

    parser = QueryParser(tables, idt, meta)
    for query in queries:
        result = parser.parse(query)
    return parser.save_history('.sql_history')


if __name__ == "__main__":
    main()
