import xml.etree.ElementTree as ET

class Synset:
    """A minimal WordNet-like Synset object."""

    def __init__(self, sid, definition, lemmas=None, pos=None):
        self._id = sid
        self._definition = definition or ""
        self._lemmas = lemmas or []
        self._pos = pos

    def definition(self):
        """Return the textual definition."""
        return self._definition

    def lemma_names(self):
        """Return list of lemma strings."""
        return self._lemmas

    def name(self):
        """Return the synset ID (like WordNet name)."""
        return self._id

    def pos(self):
        """Return the part of speech, if known."""
        return self._pos

    def __repr__(self):
        return f"Synset('{self._id}')"


class WordNet:
    """A very small WordNet-like object supporting wn.synsets() and Synset.definition()."""

    def __init__(self):
        self._synsets = {}  # synset_id -> Synset
        self._by_lemma = {}  # lemma.lower() -> [Synset]

    def synset(self, sid):
        """Return a Synset by ID."""
        return self._synsets.get(sid)

    def synsets(self, lemma=None, pos=None):
        """
        Return a list of Synset objects.

        - If lemma is None: returns all synsets.
        - If lemma is given: returns all synsets containing that lemma.
        - Optional pos filter.
        """
        if lemma is None:
            syns = list(self._synsets.values())
        else:
            syns = self._by_lemma.get(lemma.lower(), [])

        if pos is not None:
            syns = [s for s in syns if s.pos() == pos]
        return syns

    @classmethod
    def from_xml(cls, xml_path_or_str):
        """
        Load from an XML file path or XML text string.
        Returns a WordNet-like object.
        """
        self = cls()

        # Try file path first; fall back to parsing as text
        try:
            root = ET.parse(xml_path_or_str).getroot()
        except (FileNotFoundError, ET.ParseError):
            root = ET.fromstring(xml_path_or_str)

        for syn in root.findall(".//SYNSET"):
            sid = _text(syn.find("ID"))
            definition = _text(syn.find("DEF"))
            pos = _text(syn.find("POS"))

            if not sid:
                continue

            lemmas = [
                _text(l)
                for l in syn.findall("./SYNONYM/LITERAL")
                if _text(l) and _text(l) != "_EMPTY_"
            ]

            synset = Synset(sid, definition, lemmas, pos)
            self._synsets[sid] = synset

            # Index lemmas
            for lemma in lemmas:
                self._by_lemma.setdefault(lemma.lower(), []).append(synset)

        return self


def _text(el):
    return el.text.strip() if el is not None and el.text else None
