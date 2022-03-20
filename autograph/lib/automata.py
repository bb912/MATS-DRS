from typing import Set, List, Union

import networkx
from flloat.parser.ltlf import LTLfParser
from flloat.semantics.pl import _PLInterpretation
from pythomata.base import DFA

import autograph.lib.hoa as hoa


def style_agraph(network: networkx.MultiDiGraph, currentstates: List[int], layout="dot"):
    """
    Turn an automaton network into a nicely styled agraph (graphviz graph).
    :param network: The network to format
    :param currentstates: A list of states to highlight in the output
    :return: The formatted graph
    """
    graph = networkx.drawing.nx_agraph.to_agraph(network)

    for state in currentstates:
        node = graph.get_node(state)
        node.attr["style"] = "filled"
        node.attr["fillcolor"] = "yellow"

    for state in network:
        acc = network.nodes[state]["acceptance"]
        if acc and len(acc) > 0:
            # graph.get_node(state).attr["label"] = str(state) + "\n" + str(acc)
            graph.get_node(state).attr["shape"] = "doublecircle"
        else:
            graph.get_node(state).attr["shape"] = "circle"

    for u, v, key, data in network.edges(keys=True, data=True):
        graph.get_edge(u, v, key).attr["label"] = " " + data["label"] + " "

    # graph.add_node("start")
    # graph.get_node("start").attr["style"] = "invis"
    # graph.add_edge("start", network.graph["start"])

    if layout:
        graph.layout(prog=layout)

    return graph


class AutomatonSet:
    """
    An automaton, along with one or more states that the automaton is currently in.
    """

    def __init__(self, graph: "AutomatonGraph", states: Set[int]):
        self.states = states
        self.graph = graph

    @staticmethod
    def from_hoa(automaton: hoa.HOA):
        """
        Create an automaton graph from a parsed HOA file
        :param automaton: The HOA file
        :return: An automaton in the start state
        """
        graph, start = AutomatonGraph.from_hoa(automaton)
        return AutomatonSet(graph, {start})

    @staticmethod
    def from_ltlf(ltl: str, apnames: List[str]):
        graph, start = AutomatonGraph.from_ltlf(ltl, apnames)
        return AutomatonSet(graph, {start})

    def _repr_svg_(self):
        return self.graph.to_image(self.states)

    def transition(self, aps):
        """
        Transition an automaton from all the states it might be in now to all the states that
        :param aps: A bitmap of atomic propositions to determine which transitions to take
        """
        states = [self.graph.transition_states(state, aps) for state in self.states]

        return AutomatonSet(self.graph, set().union(*states))

    def acceptance(self):
        """
        Get a set of all acceptances which are currently satisfied by any state that the automaton may be in
        """
        accs = [self.graph.acceptance(state) for state in self.states]

        return set().union(*accs)

    def reaches_acceptance(self):
        return any(map(self.graph.reaches_acceptance, self.states))

    def __eq__(self, other):
        return isinstance(other, AutomatonSet) and self.states == other.states and self.graph == other.graph


def alike(tup1, tup2):
    diff = sum(0 if e1 == e2 else 1 for e1, e2 in zip(tup1, tup2))
    return diff <= 1


def combined_tuple(tup1, tup2):
    return tuple(e1 if e1 == e2 else "X" for e1, e2 in zip(tup1, tup2))


def simplify_conds(conds):
    """
    Combine multiple conditions into as few as possible.
    For example, (1, 0, 1) and (1, 0, 0) become (1, 0, X).
    This process is repeated until it doesn't do anything.
    Note that this may not be a perfect solution (np-complete problem), but it provides a decent approximation.
    :param conds:
    :return:
    """
    conds = sorted(conds)

    lastconds = None
    while lastconds != conds:
        lastconds = conds.copy()
        conds = set()
        for cond in lastconds:
            added = False

            for comp in conds:
                if alike(cond, comp):
                    conds.remove(comp)
                    conds.add(combined_tuple(cond, comp))
                    added = True
                    break

            if not added:
                conds.add(cond)

    return conds


class evaluator():
    def __init__(self, c):
        self.c = c

    def __call__(self, aps):
        return tuple(aps) in self.c


class AutomatonGraph:
    """
    Stateless representation of an automaton which may be non-deterministic
    """

    def __init__(self, network: networkx.MultiDiGraph):
        self.network = network
        accepted_states = [node for node in network.nodes if len(self.acceptance(node)) > 0]
        path_lengths = {fr: to for fr, to in networkx.all_pairs_shortest_path_length(network)}
        self.reachable = {node for node in network.nodes if
                          any(map(lambda dest: dest in path_lengths[node], accepted_states))}

    @staticmethod
    def from_hoa(automaton: hoa.HOA):
        """
        Turn a parse tree of a hoa file into an automaton and extract the starting state
        :return: The graph, and the starting state of the graph
        """
        graph = networkx.MultiDiGraph()

        start = None
        alias = dict()
        apnames = None

        # Process headers
        for headeritem in automaton.header.items:
            if isinstance(headeritem, hoa.StartHeader):  # Starting state of automaton
                start = int(headeritem.startstate)
            elif isinstance(headeritem, hoa.AliasHeader):  # Take care of aliases
                alias[headeritem.name] = headeritem.label
            elif isinstance(headeritem, hoa.APHeader):  # Names of atomic propositions
                apnames = [name.value for name in headeritem.props]

        assert start is not None, "Automaton needs a start value"
        assert apnames is not None, "Automaton needs to have at least one atomic proposition"

        # Add each individual state to graph
        for state in automaton.body:
            id = int(state.id)
            accsig = state.accsig if state.accsig else []
            accsig: Set[int] = {int(condition) for condition in accsig}
            graph.add_node(id, acceptance=accsig)

            for edge in state.edges:
                assert not edge.accsig, "Automaton must use state-based acceptance"
                to = int(edge.dest_state)
                lbl: hoa.LabelExpr = edge.label.value

                # Generate evaluator for edge (accepts array of APs and outputs bool)
                graph.add_edge(id, to, evaluator=lbl.generate_evaluator(alias), label=lbl.str_with_ap_names(apnames))

        graph.graph["start"] = start

        return AutomatonGraph(graph), start

    @staticmethod
    def from_ltlf(ltlf: str, ap_names: List[str]):
        """
        Construct an automaton graph from a DFA
        :param ltlf: The ltlf formula
        :param ap_names: An ordered list of names for the atomic propositions
        """
        ltl_parser = LTLfParser()

        # Parse to DFA
        parsed_formula = ltl_parser(ltlf)
        dfa: DFA = parsed_formula.to_automaton(determinize=True)
        assert 2 ** len(ap_names) == len(dfa.alphabet.symbols), "Alphabet size mismatch- make sure your ap_names are " \
                                                                "the right length"
        graph = networkx.MultiDiGraph()

        for state in dfa.states:
            trans_to = dfa.transition_function[state]
            states_to = dict()

            accsig = {0} if state in dfa.accepting_states else set()
            graph.add_node(state, acceptance=accsig)

            # Go from {set(atomprop):state} to {state:f_set([atomprop-bitmap])}

            # Collect {state:list[set(atomprop)]}
            for cond, dest in trans_to.items():
                if dest not in states_to:
                    states_to[dest] = [cond]
                else:
                    states_to[dest].append(cond)

            # set(atomprop) -> [atomprop-bitmap]
            def bitmap_for_conditions(cond: _PLInterpretation):
                return tuple([name in cond.true_propositions for name in ap_names])

            # Put it all together
            states_to_bitmap = {to: frozenset(map(bitmap_for_conditions, conds)) for to, conds in states_to.items()}

            for to, conds in states_to_bitmap.items():
                graph.add_edge(state, to,
                               evaluator=evaluator(conds),
                               label=str(simplify_conds(conds)).replace("\'X\'", "X").replace("True", "T").replace(
                                   "False", "F"))

        graph.graph["start"] = dfa.initial_state
        return AutomatonGraph(graph), dfa.initial_state

    def transition_states(self, state: int, aps: List[bool]):
        """
        What states can be the direct result of a given transition?
        :param state: The state that the transition starts at
        :param aps: A bitmap of atomic propositions that are True and False
        :return: A set of states that the automaton transitions to
        """
        outstates = set()

        for u, v, data in self.network.edges(state, data=True):
            label: hoa.Evaluator = data["evaluator"]
            if label(aps):
                outstates.add(int(v))

        return outstates

    def reaches_acceptance(self, state):
        return state in self.reachable

    def acceptance(self, state):
        """
        Get the acceptance conditions for a given state
        :return: A set of acceptance conditions
        """
        return self.network.nodes[state]["acceptance"]

    def to_image(self, current_state: Union[int, Set[int]] = None):
        """
        Render an AutomationGraph to an image, used for public access to _repr_svg_
        :param current_state: Either a state number or set of states that the automaton is in
        :return: An svg image
        """
        return self._repr_svg_(current_state)

    def _repr_svg_(self, current_state: Union[int, Set[int]] = None):
        if current_state is None:
            current_state = set()

        if isinstance(current_state, int):
            current_state = [current_state]

        return style_agraph(self.network, current_state).draw(format='svg').decode("utf-8")
