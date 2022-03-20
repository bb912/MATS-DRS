import pypeg2 as p
import re
from pypeg2 import attr, endl, Symbol, blank, some, maybe_some, restline, optional
from typing import List, Dict, Union, Callable

nnint = re.compile("\\d+")
insidestr = re.compile("(?:[^\"]|\"\")*")
nonws = re.compile("[^\\s:]+")
aname = re.compile("@[0-9a-zA-Z_-]+")
tf = re.compile("[tf]")

Evaluator = Callable[[List[bool]], bool]


class AutoStringable:
    def __str__(self):
        name = type(self).__name__

        return name + "(" + p.compose(self) + ")"

    def __repr__(self):
        return self.__str__()


class LabelExpr:
    def __init__(self, tokens: List[Union[str, 'LabelExpr']] = None):
        self.tokens = tokens

    def __str__(self):
        return p.compose(self, autoblank=False)

    def str_with_ap_names(self, ap_names):
        out = []
        for token in self.tokens:
            if isinstance(token, LabelExpr):
                out.append("(" + token.str_with_ap_names(ap_names) + ")")
            elif nnint.match(token):
                out.append(str(ap_names[int(token)]))
            elif token == "|":
                out.append("∨")
            elif token == "&":
                out.append("∧")
            elif token == "!":
                out.append("\u00AC")
            else:
                out.append(token)

        return "".join(out)

    # See docs of shunting_parse for more info
    def generate_evaluator(self, alias) -> Evaluator:
        func = shunting_parse(self.tokens, alias)
        return func

    # For compatibility with older test cases
    def evaluate(self, ap, context):
        return self.generate_evaluator(context["alias"])(ap)


LabelExpr.grammar = attr("tokens", some([nnint, aname, re.compile("[tf!|&]"), ("(", LabelExpr, ")")]))


class QuotedString(AutoStringable):

    def __init__(self, value: str = None):
        self.value = value

    grammar = "\"", attr("value", insidestr), "\""


class FormatVersion(AutoStringable):
    def __init__(self, version: str = None):
        self.version = version

    grammar = "HOA:", blank, attr("version", Symbol)


class StatesHeader(AutoStringable):
    def __init__(self, numstates: str = None):
        self.numstates = numstates

    grammar = "States:", blank, attr("numstates", nnint)


class StartHeader(AutoStringable):
    def __init__(self, startstate: str = None):
        self.startstate = startstate

    grammar = "Start:", blank, attr("startstate", nnint)


class APHeader(AutoStringable):
    def __init__(self, numprops: str = None, props: List[QuotedString] = None):
        self.numprops = numprops
        self.props = props

    grammar = "AP:", blank, \
              attr("numprops", nnint), blank, \
              attr("props", maybe_some((QuotedString, blank)))


class AcceptanceHeader(AutoStringable):
    def __init__(self, num: str = None, line: str = None):
        self.num = num
        self.line = line

    grammar = "Acceptance:", blank, \
              attr("num", nnint), blank, attr("line", restline)


class AliasHeader(AutoStringable):
    def __init__(self, name: str = None, label: LabelExpr = None):
        self.name = name
        self.label = label

    grammar = "Alias:", blank, \
              attr("name", aname), blank, \
              attr("label", LabelExpr)


class GenericHeader(AutoStringable):
    def __init__(self, name: str = None, value: str = None):
        self.name = name
        self.value = value

    grammar = attr("name", nonws), ":", blank, attr("value", restline)


HeaderType = Union[StatesHeader, StartHeader, APHeader, AcceptanceHeader, GenericHeader, AliasHeader]


class Header(AutoStringable):
    def __init__(self, format_version: FormatVersion = None, items: List[HeaderType] = None):
        self.format_version = format_version
        self.items = items

    grammar = attr("format_version", FormatVersion), endl, \
              attr("items", some(([StatesHeader, StartHeader, APHeader,
                                   AcceptanceHeader, GenericHeader, AliasHeader], endl)))


class AccSig(p.List):
    grammar = "{", maybe_some(nnint), "}"


class Label(AutoStringable):
    def __init__(self, value: LabelExpr = None):
        self.value = value

    grammar = "[", attr("value", LabelExpr), "]"


class Edge(AutoStringable):
    def __init__(self, label: Label = None, dest_state: str = None, accsig: AccSig = None):
        self.label = label
        self.dest_state = dest_state
        self.accsig = accsig

    grammar = attr("label", Label), blank, \
              attr("dest_state", nnint), \
              attr("accsig", optional(AccSig)), endl


class State(AutoStringable):
    def __init__(self, id: str = None, accsig: AccSig = None, edges: List[Edge] = None):
        self.id = id
        self.accsig = accsig
        self.edges = edges

    grammar = "State:", attr("id", nnint), \
              attr("accsig", optional(AccSig)), \
              endl, \
              attr("edges", maybe_some(Edge))


class Body(p.List):
    grammar = maybe_some(State)


class HOA(AutoStringable):
    def __init__(self, header: Header = None, body: Body = None):
        self.header = header
        self.body = body

    grammar = attr("header", Header), endl, \
              "--BODY--", endl, \
              attr("body", Body), endl, \
              "--END--"


# Higher precedence means "tighter" evaluation
op_precedence = {
    "!": 3,
    "&": 2,
    "|": 1
}

# How many arguments, then the way to actually evaluate it
op_meaning = {
    "!": (1, lambda b: not b),
    "&": (2, lambda a, b: a and b),
    "|": (2, lambda a, b: a or b)
}


def shunting_parse(tokens: List[Union[str, LabelExpr]], alias: Dict[str, LabelExpr]) -> Evaluator:
    """
    A modified shunting algorithm parser written in a functional style that takes advantage of the fact that parentheses
    are taken care of. Also resolves aliases.
    :param tokens: A tokenized expression, generated by LabelExpr
    :param alias: A mapping of alias names to values
    :return: A function that accepts a vector of atomic propositions to get
    """
    operator_stack = []
    output_stack = []

    def use_operator(op: str) -> None:
        # Get the required arguments from the stack
        numargs, func = op_meaning[op]
        assert (len(output_stack) >= numargs)
        argfuncs = [a for a in reversed([output_stack.pop() for i in range(numargs)])]

        # Delayed function to evaluate all arguments, then evaluate the combinator
        def combined_func(ap: List[bool]) -> bool:
            return func(*[argfunc(ap) for argfunc in argfuncs])

        output_stack.append(combined_func)

    for token in tokens:
        if isinstance(token, LabelExpr):  # Parentheses, do a recursive call
            output_stack.append(token.generate_evaluator(alias))
        elif nnint.match(token):  # The number of an atomic proposition
            output_stack.append(lambda ap, t=token: ap[int(t)])
        elif aname.match(token):  # Alias, treat similarly to parentheses + a lookup
            thisal = alias[token]
            output_stack.append(thisal.generate_evaluator(alias))
        elif token is "t":  # Always return true
            output_stack.append(lambda ap: True)
        elif token is "f":  # Always return false
            output_stack.append(lambda ap: False)
        else:  # See wikipedia Shunting Algorithm (this case is simplified since parentheses are handled already)
            while len(operator_stack) > 0 and op_precedence[token] < op_precedence[operator_stack[-1]]:
                use_operator(operator_stack.pop())
            operator_stack.append(token)

    # Apply rest of the operators
    while len(operator_stack) > 0:
        use_operator(operator_stack.pop())

    # The only thing left should be the result
    assert len(output_stack) == 1

    return output_stack[0]


def parse(text: str) -> HOA:
    return p.parse(text, HOA)
