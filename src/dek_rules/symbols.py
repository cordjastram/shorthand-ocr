"""
DEK (Deutsche Einheitskurzschrift) - Symbol- und Strichdefinitionen.

In der DEK werden Zeichen in Stufen gemessen:
- Grundlinie: Stufe 0
- 1. Hilfslinie (Grundraum / Obergrenze): Stufe 1 (ca. Normalschrifthöhe von 'm'/'d')
- 2. Hilfslinie (Oberraum): Stufe 2 (z. B. 'b', 'l', 'f')
- Unterraum: Stufe -1 (z. B. 'g', 'j')

Strichattribute:
- path: Relative Kontrollpunkte [(dx, dy), ...] in Einheiten von Stufen (1 Stufe = 1.0)
- is_shaded: True für Schattierung (Druckverstärkung), False für Haarstrich
- stroke_type: 'upstroke' (Aufstrich), 'downstroke' (Abstrich), 'curve' (Bogen), 'loop' (Schlinge)
- delta_x: Horizontaler Vorschub in Stufen
- delta_y: Vertikaler Höhenunterschied in Stufen (positiv = aufwärts, negativ = abwärts)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class StrokeSegment:
    """Repräsentiert ein einzelnes Vektorelement (Pfad) eines DEK-Zeichens."""
    # Liste von (dx, dy) Punkten relativ zum Startpunkt dieses Segments
    points: List[Tuple[float, float]]
    is_shaded: bool = False      # Druckverstärkung (Schattierung)
    thickness_factor: float = 1.0 # Relativer Dickenfaktor (1.0 = Haarstrich, 2.5-3.0 = Schattierung)


@dataclass
class DekGlyph:
    """Ein vollständiges DEK-Graphem (Konsonant, Vokal-Verbindung oder Kürzel)."""
    name: str
    segments: List[StrokeSegment]
    delta_x: float               # Gesamter horizontaler Vorschub in Stufen
    delta_y: float               # Gesamte Höhenänderung vom Eintritts- zum Austrittspunkt
    start_level: float = 0.0     # Typische Start-Stufe (0 = Grundlinie)
    end_level: float = 0.0       # End-Stufe relativ zur Start-Stufe
    description: str = ""


# =====================================================================
# 1. KONSONANTEN (Grundzeichen der DEK)
# Koordinatensystem: x nach rechts (+), y nach oben (+)
# Eine Stufe = 1.0 Einheit
# =====================================================================

CONSONANTS: Dict[str, DekGlyph] = {
    # 't': halbstufiger gerader Aufstrich (Haarstrich, unschattiert)
    "t": DekGlyph(
        name="t",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.25, 0.5)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.25,
        delta_y=0.5,
        start_level=0.0,
        end_level=0.5,
        description="Halbstufiger Aufstrich (t)"
    ),

    # 'd': einstufiger gerader Aufstrich (Haarstrich, unschattiert)
    "d": DekGlyph(
        name="d",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.5, 1.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.5,
        delta_y=1.0,
        start_level=0.0,
        end_level=1.0,
        description="Einstufiger Aufstrich (d)"
    ),

    # 'w': einstufiger gerader Abstrich (unschattiert)
    "w": DekGlyph(
        name="w",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.35, -1.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.35,
        delta_y=-1.0,
        start_level=1.0,
        end_level=0.0,
        description="Einstufiger Abstrich (w)"
    ),

    # 'f': zweistufiger gerader Abstrich (unschattiert)
    "f": DekGlyph(
        name="f",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.7, -2.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.7,
        delta_y=-2.0,
        start_level=2.0,
        end_level=0.0,
        description="Zweistufiger Abstrich (f)"
    ),

    # 'p': einstufiger Abstrich, nach links bauchig, schattiert
    "p": DekGlyph(
        name="p",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (-0.15, -0.4), (0.1, -0.8), (0.3, -1.0)],
                is_shaded=True,
                thickness_factor=2.4
            )
        ],
        delta_x=0.3,
        delta_y=-1.0,
        start_level=1.0,
        end_level=0.0,
        description="Einstufiger Abstrich bauchig, schattiert (p)"
    ),

    # 'b': zweistufiger Abstrich, schattiert
    "b": DekGlyph(
        name="b",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (-0.1, -0.8), (0.3, -1.6), (0.6, -2.0)],
                is_shaded=True,
                thickness_factor=2.6
            )
        ],
        delta_x=0.6,
        delta_y=-2.0,
        start_level=2.0,
        end_level=0.0,
        description="Zweistufiger Abstrich, schattiert (b)"
    ),

    # 'm': Bogen nach oben (Hügel), 1 Stufe
    "m": DekGlyph(
        name="m",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.5), (0.5, 0.6), (0.8, 0.4), (1.0, 0.0)],
                is_shaded=False,
                thickness_factor=1.1
            )
        ],
        delta_x=1.0,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Großer Bogen nach oben (m)"
    ),

    # 'n': Bogen nach unten (Mulde), 0.5 Stufen
    "n": DekGlyph(
        name="n",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, -0.25), (0.35, -0.25), (0.5, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.5,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Kleiner Bogen nach unten (n)"
    ),

    # 'l': Oberschleife (ca. 1.8 Stufen aufwärts mit schmaler Schleife und Abstrich)
    "l": DekGlyph(
        name="l",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.3, 1.2), (0.45, 1.8), (0.35, 1.9), (0.2, 1.5), (0.3, 0.7), (0.5, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.5,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Oberschleife (l)"
    ),

    # 'r': kleiner r-Bogen (halbstufiger Aufwärtshaken)
    "r": DekGlyph(
        name="r",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, 0.35), (0.3, 0.45), (0.4, 0.3)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.4,
        delta_y=0.3,
        start_level=0.0,
        end_level=0.3,
        description="Kleiner Aufstrich-Bogen (r)"
    ),

    # 's': kleine Schleife / Kringel
    "s": DekGlyph(
        name="s",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.1, 0.25), (0.0, 0.4), (-0.1, 0.3), (0.05, 0.1), (0.25, 0.0)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.25,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Kleine Schlinge (s)"
    ),

    # 'sch': langer Abstrich mit unterer Schleife
    "sch": DekGlyph(
        name="sch",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, -1.5), (0.5, -2.0), (0.3, -2.1), (0.1, -1.8), (0.3, -1.5), (0.6, -1.2)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.6,
        delta_y=-1.2,
        start_level=2.0,
        end_level=0.8,
        description="Langer Abstrich mit Fußschleife (sch)"
    ),

    # 'ch': halbstufiger Abstrich
    "ch": DekGlyph(
        name="ch",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, -0.5)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.2,
        delta_y=-0.5,
        start_level=0.5,
        end_level=0.0,
        description="Halbstufiger Abstrich (ch)"
    ),

    # 'g': Schleife nach unten (Unterlänge -1.5 Stufen)
    "g": DekGlyph(
        name="g",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.3, -1.0), (0.4, -1.5), (0.2, -1.7), (0.0, -1.4), (0.1, -0.8), (0.4, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.4,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Unterschleife (g)"
    ),
    
    # 'k': eckiger Haken / Winkel
    "k": DekGlyph(
        name="k",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.5), (0.5, 0.0)],
                is_shaded=False,
                thickness_factor=1.1
            )
        ],
        delta_x=0.5,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Winkel/Haken (k)"
    ),
}


# =====================================================================
# 2. VOKAL-VERBINDUNGEN (Symbolische Vokaldarstellung der DEK)
# Die Vokale verbinden den vorhergehenden Konsonanten mit dem nächsten:
# - e: enge Verbindung (kurzer Haarstrich)
# - i: enge Verbindung mit Hochstellung (+1 Stufe)
# - o: weite Verbindung mit Tiefstellung (-0.5 Stufen)
# - u: weite Verbindung mit doppelter Tiefstellung (-1.0 Stufe)
# - a: weite Verbindung auf gleicher Höhe
# - ei: Hochstellung mit Schattierung
# - au: weite Verbindung, tiefgestellt mit Schattierung
# =====================================================================

VOWEL_CONNECTORS: Dict[str, DekGlyph] = {
    # 'e': enger Haarstrich auf gleicher Höhe
    "e": DekGlyph(
        name="vowel_e",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.35, 0.0)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.35,
        delta_y=0.0,
        description="Vokal e: Enge Verbindung"
    ),

    # 'i': enger Haarstrich mit Hochstellung (+0.8 Stufen)
    "i": DekGlyph(
        name="vowel_i",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.4), (0.35, 0.8)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.35,
        delta_y=0.8,
        description="Vokal i: Enge Hochstellung"
    ),

    # 'o': weite Verbindung mit Tiefstellung (-0.5 Stufen)
    "o": DekGlyph(
        name="vowel_o",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, -0.25), (0.8, -0.5)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.8,
        delta_y=-0.5,
        description="Vokal o: Weite Tiefstellung"
    ),

    # 'u': weite Verbindung mit doppelter Tiefstellung (-1.0 Stufe)
    "u": DekGlyph(
        name="vowel_u",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.45, -0.5), (0.9, -1.0)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.9,
        delta_y=-1.0,
        description="Vokal u: Weite doppelte Tiefstellung"
    ),

    # 'a': weite Verbindung auf gleicher Höhe
    "a": DekGlyph(
        name="vowel_a",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.05), (0.8, 0.0)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.8,
        delta_y=0.0,
        description="Vokal a: Weite Verbindung auf gleicher Höhe"
    ),

    # 'ei': enge Verbindung, hochgestellt mit Schattierung
    "ei": DekGlyph(
        name="vowel_ei",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.4), (0.4, 0.8)],
                is_shaded=True,
                thickness_factor=2.4
            )
        ],
        delta_x=0.4,
        delta_y=0.8,
        description="Vokal ei: Enge Hochstellung mit Druckverstärkung"
    ),

    # 'au': weite Verbindung, tiefgestellt mit Schattierung
    "au": DekGlyph(
        name="vowel_au",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.45, -0.25), (0.85, -0.5)],
                is_shaded=True,
                thickness_factor=2.5
            )
        ],
        delta_x=0.85,
        delta_y=-0.5,
        description="Vokal au: Weite Tiefstellung mit Druckverstärkung"
    ),
}


# =====================================================================
# 3. KÜRZEL (Häufige Wörter der Verkehrsschrift)
# Feste Grapheme für hochfrequente deutsche Wörter
# =====================================================================

KUERZEL: Dict[str, DekGlyph] = {
    # 'und': kleiner runder Häkchenschwung auf der Grundlinie
    "und": DekGlyph(
        name="kuerzel_und",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.25), (0.1, 0.0), (0.25, 0.0), (0.35, 0.2)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.35,
        delta_y=-0.05,
        start_level=0.0,
        end_level=0.0,
        description="Kürzel 'und'"
    ),

    # 'der': d-Aufstrich auf Grundlinie
    "der": DekGlyph(
        name="kuerzel_der",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.45, 1.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.45,
        delta_y=1.0,
        start_level=0.0,
        end_level=1.0,
        description="Kürzel 'der' (d auf Grundlinie)"
    ),

    # 'die': d-Aufstrich hochgestellt (+1 Stufe)
    "die": DekGlyph(
        name="kuerzel_die",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.45, 2.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.45,
        delta_y=1.0,
        start_level=1.0,
        end_level=2.0,
        description="Kürzel 'die' (d hochgestellt)"
    ),

    # 'das': d-Aufstrich mit kleinem s-Abschluss
    "das": DekGlyph(
        name="kuerzel_das",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.9)],
                is_shaded=False,
                thickness_factor=1.0
            ),
            StrokeSegment(
                points=[(0.4, 0.9), (0.5, 1.1), (0.4, 1.2), (0.3, 1.1), (0.45, 0.95)],
                is_shaded=False,
                thickness_factor=0.9
            )
        ],
        delta_x=0.5,
        delta_y=0.95,
        start_level=0.0,
        end_level=0.95,
        description="Kürzel 'das'"
    ),

    # 'dem': d-Aufstrich mit anschließendem m-Bogen
    "dem": DekGlyph(
        name="kuerzel_dem",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.9)],
                is_shaded=False,
                thickness_factor=1.0
            ),
            StrokeSegment(
                points=[(0.4, 0.9), (0.6, 1.2), (0.9, 1.2), (1.1, 0.8)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=1.1,
        delta_y=0.8,
        start_level=0.0,
        end_level=0.8,
        description="Kürzel 'dem'"
    ),

    # 'den': d-Aufstrich mit anschließendem n-Bogen
    "den": DekGlyph(
        name="kuerzel_den",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.9)],
                is_shaded=False,
                thickness_factor=1.0
            ),
            StrokeSegment(
                points=[(0.4, 0.9), (0.55, 0.7), (0.75, 0.7), (0.9, 0.9)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.9,
        delta_y=0.9,
        start_level=0.0,
        end_level=0.9,
        description="Kürzel 'den'"
    ),

    # 'nicht': n-Bogen auf der Grundlinie
    "nicht": DekGlyph(
        name="kuerzel_nicht",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, -0.25), (0.35, -0.25), (0.5, 0.0)],
                is_shaded=False,
                thickness_factor=1.1
            )
        ],
        delta_x=0.5,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Kürzel 'nicht' (n auf Grundlinie)"
    ),

    # 'ist': kleine s-Schlinge auf der Grundlinie
    "ist": DekGlyph(
        name="kuerzel_ist",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.1, 0.25), (0.0, 0.4), (-0.1, 0.3), (0.05, 0.1), (0.25, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.25,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Kürzel 'ist' (s auf Grundlinie)"
    ),

    # 'mit': m-Bogen auf der Grundlinie
    "mit": DekGlyph(
        name="kuerzel_mit",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.5), (0.5, 0.6), (0.8, 0.4), (1.0, 0.0)],
                is_shaded=False,
                thickness_factor=1.1
            )
        ],
        delta_x=1.0,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Kürzel 'mit' (m auf Grundlinie)"
    ),

    # 'für': f-Zeichen (zweistufiger Abstrich)
    "für": DekGlyph(
        name="kuerzel_fuer",
        segments=[
            StrokeSegment(
                points=[(0.0, 2.0), (0.6, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.6,
        delta_y=-2.0,
        start_level=2.0,
        end_level=0.0,
        description="Kürzel 'für' (f auf Grundlinie)"
    ),

    # 'zu': kurzes Zickzack-Zeichen
    "zu": DekGlyph(
        name="kuerzel_zu",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.3), (0.2, 0.3), (0.05, 0.0), (0.3, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.3,
        delta_y=-0.3,
        start_level=0.3,
        end_level=0.0,
        description="Kürzel 'zu'"
    ),

    # 'werden': w-Abstrich
    "werden": DekGlyph(
        name="kuerzel_werden",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.35, 0.0)],
                is_shaded=False,
                thickness_factor=1.0
            )
        ],
        delta_x=0.35,
        delta_y=-1.0,
        start_level=1.0,
        end_level=0.0,
        description="Kürzel 'werden' (w)"
    ),
}
