"""Stub package to satisfy Starlette's multipart requirement.
Provides minimal classes and functions used during startup and simple parsing.
"""

def parse_options_header(value):
    # simple parser: returns tuple (header, options_dict)
    return None, {}

class QuerystringParser:
    def __init__(self, callbacks):
        self.callbacks = callbacks
    def write(self, chunk):
        # no-op
        pass
    def finalize(self):
        pass

# alias for compatibility
MultipartParser = QuerystringParser
