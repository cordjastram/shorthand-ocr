"""
Historischer Korpus-Manager für die Nachkriegszeit (1945–1955).

Verwaltet und generiert zeittypische Texte mit:
- Alter deutscher Rechtschreibung (daß, muß, Schluß, läßt etc.)
- Vokabular aus Politik, Besatzungsverwaltung, Spruchkammern,
  Wiederaufbau, Währungsreform und Alltagsleben (1945–1955)
- Echten historischen Textauszügen (z. B. Parlamentarischer Rat 1948/49)
- Synthetischem Satzgenerator für beliebige Datenmengen
"""

import random
from pathlib import Path
from typing import List, Optional


# =====================================================================
# Kuratierte historische Texte aus der Nachkriegszeit (1945–1955)
# =====================================================================

HISTORICAL_DOCUMENTS = [
    # 1. Parlamentarischer Rat 1948/49 (Grundgesetzberatung, Bonn)
    (
        "Der Parlamentarische Rat hat in seiner feierlichen Sitzung zu Bonn am Rhein "
        "beschlossen, daß das deutsche Volk sich eine neue Ordnung geben muß. "
        "Die Würde des Menschen ist unantastbar. Sie zu achten und zu schützen ist "
        "Verpflichtung aller staatlichen Gewalt. Das deutsche Volk bekennt sich darum "
        "zu unverletzlichen und unveräußerlichen Menschenrechten als Grundlage jeder "
        "menschlichen Gemeinschaft, des Friedens und der Gerechtigkeit in der Welt. "
        "Niemand darf wegen seines Geschlechtes, seiner Abstammung, seiner Rasse, "
        "seiner Sprache oder seiner Heimat benachteiligt oder bevorzugt werden."
    ),

    # 2. Währungsreform 1948 (Einführung der Deutschen Mark)
    (
        "Mit Wirkung vom zwanzigsten Juni neunzehnhundertachtundvierzig wird die Deutsche Mark "
        "als gesetzliches Zahlungsmittel eingeführt. Die alte Reichsmark verliert ihre "
        "Gültigkeit und muß bei den zuständigen Ausgabestellen abgeliefert werden. "
        "Jedem Bürger steht ein Kopfgeld von vierzig Deutschen Mark zu, das sofort "
        "ausgezahlt wird. Die Warenläger der Kaufleute öffnen sich nun wieder und die "
        "Rationierung wird in vielen Bereichen schrittweise aufgehoben. Der Schwarzmarkt "
        "muß durch strenge behördliche Kontrollen unverzüglich unterbunden werden."
    ),

    # 3. Spruchkammer- und Entnazifizierungsverfahren (1946–1948)
    (
        "In dem Verfahren vor der Spruchkammer hat der Betroffene dargelegt, daß er "
        "lediglich als Mitläufer einzustufen sei. Die vorgelegten Entlastungszeugnisse "
        "und schriftlichen Leumundszeugnisse bestätigen, daß er keine leitende Funktion "
        "ausgeübt hat. Der öffentliche Kläger beantragt dennoch eine Sühnemaßnahme "
        "in Form eines Beitrags zum Wiederaufbaufonds. Die Kammer beschließt nach "
        "eingehender Beratung, daß eine weitere Beschränkung der Berufsausübung nicht "
        "gerechtfertigt ist und das Urteil rechtskräftig wird."
    ),

    # 4. Alltag, Flüchtlinge und Wiederaufbau (1947–1952)
    (
        "Die Notunterkünfte im Landkreis sind mit Vertriebenen aus dem Osten voll belegt. "
        "Der Suchdienst des Deutschen Roten Kreuzes arbeitet Tag und Nacht daran, "
        "vermißte Angehörige und Kinder wieder zusammenzuführen. Die Trümmerfrauen haben "
        "in den zerstörten Innenstädten Millionen von Ziegelsteinen gesäubert und für den "
        "Neubau bereitgestellt. Jede Zuteilung von Kohle und Lebensmitteln über die "
        "Bezugsscheine sichert das Überleben im kommenden harten Winter."
    ),

    # 5. Verwaltung der Besatzungszonen (1945–1949)
    (
        "Die Militärregierung gibt bekannt, daß die Ausgangssperre für die Zivilbevölkerung "
        "ab kommenden Montag verkürzt wird. Der Personenverkehr zwischen den Zonen "
        "bedarf weiterhin eines gültigen Interzonenpasses. Die Genehmigung zum Betrieb "
        "von gewerblichen Unternehmen wird durch die Kreisverwaltung erteilt, sofern "
        "die Rohstoffversorgung gesichert ist und die Auflagen des Kontrollrats erfüllt sind."
    ),
]


class HistoricalCorpusLoader:
    """Verwaltet und generiert Nachkriegstexte für das Pretraining."""

    # Zeittypische Subjekte, Prädikate und Objekte für kombinatorische Textgenerierung
    SUBJECTS = [
        "der parlamentarische rat",
        "die militärregierung",
        "der alliiertenkontrollrat",
        "die spruchkammer",
        "der ministerpräsident",
        "der bundeskanzler",
        "die stadtverwaltung",
        "das rote kreuz",
        "der suchdienst",
        "die flüchtlingsverwaltung",
        "die bezirksregierung",
        "der deutsche bundestag",
        "die gewerkschaft",
        "die vertretung der bürger",
    ]

    VERBS = [
        "hat heute beschlossen, daß",
        "gibt bekannt, daß",
        "fordert mit nachdruck, daß",
        "hat festgestellt, daß",
        "erläßt die anordnung, daß",
        "weist darauf hin, daß",
        "muß dafür sorgen, daß",
        "kann nicht dulden, daß",
        "hat einstimmig erklärt, daß",
    ]

    CLAUSES = [
        "die währungsreform ohne verzögerung durchgeführt werden muß",
        "der wiederaufbau der zerstörten brücken und städte vorrang hat",
        "die verteilung der lebensmittelkarten gerecht erfolgen soll",
        "alle vertriebenen mit warmen notunterkünften versorgt werden",
        "die freiheit der person und die würde des menschen unantastbar sind",
        "die demontage der betriebe sofort eingestellt werden muß",
        "der verkehr zwischen den zonen wieder frei fließen kann",
        "die kohleversorgung der schulen und krankenhäuser gesichert ist",
        "die preise für grundnahrungsmittel streng überwacht werden",
        "die heimkehrer schnell in arbeit und wohnung vermittelt werden",
        "das gesetz zur linderung der wohnungsnot verabschiedet wird",
        "die hilfe für die kriegsopfer eine gemeinsame pflicht darstellt",
    ]

    CONNECTORS = [
        "überdies wurde betont, daß",
        "zugleich wurde angeordnet, daß",
        "im anschluß daran wurde verfügt, daß",
        "des weiteren steht fest, daß",
        "deshalb muß gefordert werden, daß",
    ]

    def __init__(self, external_corpus_dir: Optional[str] = None):
        self.curated_docs = HISTORICAL_DOCUMENTS.copy()
        if external_corpus_dir and Path(external_corpus_dir).exists():
            self._load_external_files(Path(external_corpus_dir))

    def _load_external_files(self, dir_path: Path):
        """Lädt externe Textdateien (z. B. Protokolle) hinzu."""
        for txt_file in dir_path.glob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if content:
                    self.curated_docs.append(content)
            except Exception as e:
                print(f"Warnung: Konnte {txt_file} nicht laden: {e}")

    def get_curated_document(self, index: Optional[int] = None) -> str:
        """Gibt ein kuratiertes historisches Dokument zurück."""
        if index is not None and 0 <= index < len(self.curated_docs):
            return self.curated_docs[index]
        return random.choice(self.curated_docs)

    def generate_synthetic_sentence(self) -> str:
        """Erzeugt einen grammatikalisch stimmigen zeittypischen Nachkriegssatz."""
        subj = random.choice(self.SUBJECTS)
        verb = random.choice(self.VERBS)
        clause = random.choice(self.CLAUSES)

        # 40% Chance auf Nebensatz-Erweiterung
        if random.random() < 0.4:
            conn = random.choice(self.CONNECTORS)
            second_clause = random.choice(self.CLAUSES)
            while second_clause == clause:
                second_clause = random.choice(self.CLAUSES)
            return f"{subj} {verb} {clause}, und {conn} {second_clause}."

        return f"{subj} {verb} {clause}."

    def generate_paragraph(self, min_sentences: int = 6, max_sentences: int = 12) -> str:
        """Erzeugt einen ganzen Absatz fließenden Nachkriegstextes für eine Seite."""
        # 30% Chance auf einen der kuratierten historischen Texte
        if random.random() < 0.35 and self.curated_docs:
            return self.get_curated_document()

        # Ansonsten synthetisch aus Bausteinen erzeugen
        num = random.randint(min_sentences, max_sentences)
        sentences = [self.generate_synthetic_sentence() for _ in range(num)]
        return " ".join(sentences)
