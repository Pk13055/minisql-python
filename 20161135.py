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

    @staticmethod
    def parse_joins(joins: list) -> list:
        """Parse join logicals
        :param joins: list -> Keyword joins
        :return tokens: list -> simplified joins
        """
        join_tk = [_.normalized for _ in joins if _.ttype == sqlparse.tokens.Keyword]
        tokens = []
        if len(join_tk) == len(joins):
            tokens = join_tk
        else:
            raise Exception(f"joinerror: invalid logical used for joining")
        return tokens

    def parse_conditions(self, tokens: list) -> list:
        """Parses individual conditions for comparisons
        :param tokens: list of comparison tokens
        :return simple_tokens: list -> basic set of comparisons. Each object
            has a standard format
            [
                {
                    'type' : 'JOIN' OR 'NUM'
                    'lhs' : [None or 'table', 'col'],
                    'rhs' : [None or 'table', 'col'] OR NUM,
                    'op'  : 'LT', 'GT', 'EQ', 'LTE', 'GTE'
                }, ...
            ]
        """
        def parse_condition(token_st):
            cur_token = {}
            is_first = True
            for token in token_st:
                if token.ttype == sqlparse.tokens.Whitespace:
                    continue
                if token.ttype == sqlparse.tokens.Comparison:
                    cur_token['op'] = token.normalized
                elif isinstance(token, sqlparse.sql.Identifier):
                    value = self.parse_col(token.normalized)
                    cur_token['rhs' if 'lhs' in cur_token else 'lhs'] = value
                    cur_token['type'] = 'JOIN'
                    is_first = False
                elif token.ttype in [sqlparse.tokens.Token.Literal.Number.Integer,
                                     sqlparse.tokens.Token.Literal.Number.Float]:
                    cur_token['rhs'] = float(token.normalized)
                    cur_token['type'] = 'NUM'
            return cur_token

        simple_comp = [parse_condition(token) for token in tokens]
        return simple_comp

    def parse_where(self, base: list) -> list:
        """Parses the where section to extract join params

        :param base: list -> Base set of tokens to parse from
        This contains the raw where statement
        :return where_sec: dict -> where part of the `simplified_tokens`
        that contains any possible comparisons
        """
        where_sec = {}
        tokens = [_ for _ in base if _.ttype != sqlparse.tokens.Whitespace][1:-1]
        if len(tokens) < 4:
            comparisons = tokens[::2]
            joins = tokens[1::2]
            where_sec['conditions'] = self.parse_conditions(comparisons)
            where_sec['joins'] = self.parse_joins(joins)
        else:
            raise Exception("whereerror: unsupported where statement")
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
        if fields[0].ttype == sqlparse.tokens.Token.Wildcard:
            simplified_tokens['fields'] = [[None, '*']]
        else:
            fields = [_ for ls in fields for _ in ls if _.ttype not in
                    [sqlparse.tokens.Whitespace, sqlparse.tokens.Punctuation]]
            simplified_tokens['fields'] = self.parse_fields(fields, valid_columns)

        # parse where section if exists
        if last_token.ttype != sqlparse.tokens.Punctuation:
            simplified_tokens['where'] = self.parse_where(last_token)
        return simplified_tokens

    def normalize_col(self, val, rev=False):
        table, col = val
        cur_token = []
        if col == "*":
            cur_token = [None]
        elif table is None:
            valid_tables = [self.tables[_] for _ in self.idt[col]]
            cur_token.append(valid_tables[0 if not rev else -1])
            if len(valid_tables) == 0:
                raise Exception(f"colerror: invalid column referenced")
        else:
            cur_token.append(table)
        cur_token.append(col)
        return cur_token

    def normalize_fields(self, fields: list, tables: list) -> list:
        """Normalizes fields and adds the column name as well

        :param fields: list -> list of fields
        :param tables: list -> list of tables
        :return norm_list: list -> same format as previous with everything filled
        """
        norm_tokens = []
        keywords = ['sum', 'min', 'max', 'avg', 'DISTINCT']
        keywords += [_.upper() for _ in keywords]
        for col in fields:
            cur_token = []
            if any(isinstance(_, list) for _ in col):
                keyword, val = col
                if keyword not in keywords:
                    raise Exception(f"functionerror: unsupported function used")
                cur_token.append(keyword)
                cur_token += self.normalize_col(val)
            else:
                cur_token = [None] + self.normalize_col(col)
            norm_tokens.append(tuple(cur_token))
        return norm_tokens

    def normalize_where(self, tokens: list, tables: list) -> list:
        """Normalize the where and add relevant tables as well

        :param tokens: list -> list of basic tokens
        :param tables: list -> list of tables
        :return norm_tokens: list -> list of normalized table names
        """
        norm_tokens = []
        for token in tokens['conditions']:
            cur_token = token
            cur_token['lhs'] = self.normalize_col(token['lhs'])
            if token['type'] == 'JOIN':
                cur_token['rhs'] = self.normalize_col(token['rhs'], True)
            norm_tokens.append(cur_token)
        tokens['conditions'] = norm_tokens
        return tokens

    def gen_result(self, fields: list, where: dict, dataset: list, tables: list) -> dict:
        """Generate the final result query

        :param fields: list -> list of normalized fields
        :param where: None or dict -> where condition to process
        :param dataset: list -> dataset to process on
        :return res: str -> final result
        """
        queryset = {}
        result = {}
        for data_gp, table in zip(dataset, tables):
            cols = self.meta[table]
            queryset[table] = {col: [] for col in cols}
            for row in data_gp:
                [queryset[table][col].append(int(val)) for col, val in zip(cols, row)]
        for field in fields:
            keyword, table, col = field
            if col == "*":
                for table in tables:
                    for col in queryset[table]:
                        result[table + '.' + col] = queryset[table][col]
                break
            base_col = queryset[table][col]
            if keyword == "DISTINCT":
                result[f"{keyword}({table}.{col})"] = base_col
            elif keyword and keyword.lower() in ['min', 'max', 'avg', 'sum']:
                fn = eval(keyword)
                query_val = fn(base_col)
                idxs = [idx for idx, _ in enumerate(base_col) if _ == query_val]
                result[f"{keyword}({table}.{col})"] = base_col
            else:
                result[table + '.' + col] = base_col
        return result

    def filter_where(self, dataset: list, where: dict) -> list:
        """Filter the where joins and reduce data size

        :param dataset: list -> complete dataset
        :param where: dict -> the where condition to be run
        :return dataset: list -> reduced rows
        """
        new_data = []
        new_data = dataset  # TODO: filter on the basis of conditionals
        return new_data

    def print_result(self, json: dict, fields: list, verbose=True) -> list:
        """Prints the result given a formatted dict"""
        res_ = []
        header = []
        for field in fields:
            keyword, table, col = field
            if col == "*":
                res_ = [json[k] for k in json]
                header = list(json.keys())
                break
            if keyword is not None:
                header.append(f"{keyword}({table}.{col})")
            else:
                header.append(f"{table}.{col}")
            res_.append(json[header[-1]])
        res_ = list(map(list, zip(*res_)))
        header_string = '.'.join(header)
        body = ""
        for row in res_[1:]:
            body += "\n" + str(row)[1:-1]
        res = header_string + body
        if verbose:
            print(res)
        return res

    def compile_result(self, tokens: list) -> str:
        """Compile the results after verifying valid statement
        Handle the where condition and comparators and actual data fetching

        :param tokens: list -> simplified list of tokens to parse
        :return res: str -> Resulting formatted result depending on the query
        """
        valid_tables = self.tables
        idt = self.idt
        tables, fields = tokens['tables'], tokens['fields']
        fields = self.normalize_fields(fields, tables)
        where = None
        table_fetch = [self.tables.index(_) for _ in tables]
        dialects = [(t_name, self.dialects[_]) for t_name, _ in zip(tables, table_fetch)]
        readers = [csv.reader(open(t_name + '.csv'), dialect=dialect) for t_name, dialect in dialects]
        dataset = list(map(lambda x: list(x), readers))
        if 'where' in tokens:
            where = self.normalize_where(tokens['where'], tables)
            dataset = self.filter_where(dataset, where)
        try:
            result = self.gen_result(fields, where, dataset, tables)
        except Exception as e:
            print(f"dataerror: invalid query | {e}")
        return self.print_result(result, fields)

    def parse(self, query: sqlparse.sql.Statement) -> str:
        """Parses a single query and returns the result.
        Optionally, the result is also stored in the history

        :param query: sqlparse statment -> The query to be parsed
        :return result: str -> the record result to be parsed
        """
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
        self.result = res
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
