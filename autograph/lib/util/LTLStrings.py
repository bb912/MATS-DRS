class XType:
    """
    Usage: X[3]("ap") generates XXX(ap) (ap will be true 3 turns from now)
    """

    def __getitem__(self, num):
        def fun(text):
            open_xs = "X(" * num
            close_parens = ")" * num
            return open_xs + text + close_parens

        return fun


class SliceLTLType:
    def __init__(self, sep):
        self.sep = sep

    def __getitem__(self, slc: slice):
        assert slc.step is None or slc.step >= 1

        def fun(text):
            step = slc.step if slc.step else 1

            slc_len = slc.stop - slc.start
            if slc_len <= 0:
                return text
            elif slc.start > 0:
                return X[slc.start](self[0:slc_len:step](text))
            else:
                return "(" + text + ")" + self.sep + \
                       "(" + X[step](self[0:slc.stop - step:step](text)) + ")"

        return fun


X = XType()
F = SliceLTLType("|")
G = SliceLTLType("&")
