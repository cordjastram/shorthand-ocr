"""
Gabelsberger-Kurzschrift (1834) - Symbol-, Vokal- und Sigeldefinitionen.

Charakteristika des Gabelsberger-Systems (Franz Xaver Gabelsberger, 1834):
- Stark kursive, fließend-organische Zeichenformen ("Redezeichenkunst")
- 1-Linien-Grundliniensystem mit ausgeprägten Ober- und Unterlängen
- Vokalisation durch Stellung, Druck (Schattierung) und Verschmelzung im An-/Auslaut:
  * 'e': enge Verbindung auf der Grundlinie
  * 'a': Druckverstärkung (Schattierung) und mittlere Weite
  * 'i': Steilstellung / Hochstellung
  * 'o': runde Auswölbung (Wölbung nach unten/außen)
  * 'u': Tiefstellung bzw. flache Unterwölbung
- Klassische Sigel (Kürzel) für hochfrequente Wörter und Vor-/Nachsilben
"""

from typing import Dict
from src.dek_rules.symbols import DekGlyph, StrokeSegment


# =====================================================================
# 1. KONSONANTEN (Grundzeichen nach Gabelsberger)
# =====================================================================

GABELSBERGER_CONSONANTS: Dict[str, DekGlyph] = {
    # 't': langer, gerader Aufstrich (2 Stufen hoch, charakteristisch für Gabelsberger)
    "t": DekGlyph(
        name="gab_t",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.55, 1.85)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.55,
        delta_y=1.85,
        start_level=0.0,
        end_level=1.85,
        description="Gabelsberger t: Langer steiler Aufstrich",
    ),

    # 'd': weicher, leicht geschwungener einstufiger Abstrich mit kleinem Fußbogen
    "d": DekGlyph(
        name="gab_d",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.9), (0.15, 0.3), (0.3, 0.0), (0.45, 0.1)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.45,
        delta_y=-0.8,
        start_level=0.9,
        end_level=0.1,
        description="Gabelsberger d: Geschwungener Abstrich mit Fußrundung",
    ),

    # 'b': einstufiger, nach links gewölbter Bauchabstrich (leicht schattiert)
    "b": DekGlyph(
        name="gab_b",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (-0.12, 0.5), (0.05, 0.1), (0.35, 0.0), (0.5, 0.15)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.5,
        delta_y=-0.85,
        start_level=1.0,
        end_level=0.15,
        description="Gabelsberger b: Linksgewölbter Schattierungsbogen",
    ),

    # 'p': tiefer gezogener Bogenschwung mit Unterlänge und Schleifenansatz
    "p": DekGlyph(
        name="gab_p",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.2, -0.4), (0.25, -1.1), (0.05, -1.2), (0.0, -0.7), (0.4, 0.0)],
                is_shaded=True,
                thickness_factor=2.2,
            )
        ],
        delta_x=0.4,
        delta_y=-0.8,
        start_level=0.8,
        end_level=0.0,
        description="Gabelsberger p: Unterlängen-Schleifenabstrich",
    ),

    # 'w': geschmeidiger Rechtsbogen (1 Stufe)
    "w": DekGlyph(
        name="gab_w",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.9), (0.25, 0.7), (0.35, 0.3), (0.2, 0.0), (0.4, 0.05)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.4,
        delta_y=-0.85,
        start_level=0.9,
        end_level=0.05,
        description="Gabelsberger w: Geschwungener Halbbogen",
    ),

    # 'f': langer durchgehender Abstrich (Ober- in Unterlänge)
    "f": DekGlyph(
        name="gab_f",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.4), (0.35, 0.0), (0.6, -1.1), (0.75, -0.8)],
                is_shaded=True,
                thickness_factor=2.0,
            )
        ],
        delta_x=0.75,
        delta_y=-2.2,
        start_level=1.4,
        end_level=-0.8,
        description="Gabelsberger f: Durchgezogener Schrägabstrich",
    ),

    # 'm': doppelwelliger geschwungener Bogen auf der Grundlinie
    "m": DekGlyph(
        name="gab_m",
        segments=[
            StrokeSegment(
                points=[
                    (0.0, 0.0),
                    (0.2, 0.55),
                    (0.4, 0.1),
                    (0.6, 0.55),
                    (0.85, 0.0),
                ],
                is_shaded=False,
                thickness_factor=1.1,
            )
        ],
        delta_x=0.85,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Gabelsberger m: Doppelarkade",
    ),

    # 'n': flacher, wellenförmiger Bogen
    "n": DekGlyph(
        name="gab_n",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.18, 0.4), (0.35, 0.0), (0.5, 0.1)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.5,
        delta_y=0.1,
        start_level=0.0,
        end_level=0.1,
        description="Gabelsberger n: Kurze Wellenlinie",
    ),

    # 'l': sehr kleine, punktförmige Schlinge auf der Grundlinie (charakteristisch für Gabelsberger!)
    "l": DekGlyph(
        name="gab_l",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.12, 0.3), (0.02, 0.42), (-0.08, 0.25), (0.08, 0.0), (0.25, 0.05)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.25,
        delta_y=0.05,
        start_level=0.0,
        end_level=0.05,
        description="Gabelsberger l: Kleine Schlinge im Grundraum",
    ),

    # 'r': kleiner nach links geneigter Haken/Abstrich im Grundraum
    "r": DekGlyph(
        name="gab_r",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.45), (0.15, 0.0), (0.32, 0.15)],
                is_shaded=False,
                thickness_factor=1.05,
            )
        ],
        delta_x=0.32,
        delta_y=-0.3,
        start_level=0.45,
        end_level=0.15,
        description="Gabelsberger r: Kurzer Hakenabstrich",
    ),

    # 's': runder Kreisbogen / Ringel im Grundraum
    "s": DekGlyph(
        name="gab_s",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.18, 0.35), (0.05, 0.6), (-0.1, 0.35), (0.1, 0.0), (0.3, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.3,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Gabelsberger s: Rundschlinge",
    ),

    # 'sch': große geschwungene Unterlängenschleife
    "sch": DekGlyph(
        name="gab_sch",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.25, -0.6), (0.35, -1.5), (0.15, -1.65), (0.0, -1.2), (0.35, -0.3), (0.6, 0.0)],
                is_shaded=True,
                thickness_factor=1.9,
            )
        ],
        delta_x=0.6,
        delta_y=-0.8,
        start_level=0.8,
        end_level=0.0,
        description="Gabelsberger sch: Geschwungene Unterlängenschleife",
    ),

    # 'ch': weiter, von oben herabgezogener Bogen
    "ch": DekGlyph(
        name="gab_ch",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.1), (0.25, 0.9), (0.3, 0.3), (0.15, 0.0), (0.45, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.45,
        delta_y=-1.1,
        start_level=1.1,
        end_level=0.0,
        description="Gabelsberger ch: Weicher Oberbogen",
    ),

    # 'g': gewölbter Bogen im Grundraum mit Schattierung
    "g": DekGlyph(
        name="gab_g",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, 0.65), (0.4, 0.7), (0.55, 0.1), (0.65, 0.0)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.65,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Gabelsberger g: Schattierter Wölbbogen",
    ),

    # 'k': kleiner, flacher Oberbogen ohne Schattierung
    "k": DekGlyph(
        name="gab_k",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, 0.45), (0.35, 0.45), (0.5, 0.0)],
                is_shaded=False,
                thickness_factor=1.1,
            )
        ],
        delta_x=0.5,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Gabelsberger k: Kleiner K-Bogen",
    ),

    # 'z': steiler Abstrich mit kleiner Fußschlinge
    "z": DekGlyph(
        name="gab_z",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.1), (0.25, 0.0), (0.15, -0.2), (0.05, 0.0), (0.4, 0.1)],
                is_shaded=True,
                thickness_factor=2.0,
            )
        ],
        delta_x=0.4,
        delta_y=-1.0,
        start_level=1.1,
        end_level=0.1,
        description="Gabelsberger z: Schattierter Abstrich mit Fußschlinge",
    ),
}


# =====================================================================
# 2. VOKAL-VERBINDUNGEN (Gabelsberger Vokalisation)
# =====================================================================

GABELSBERGER_VOWELS: Dict[str, DekGlyph] = {
    # 'e': kurzer, flacher Verbindungsstrich
    "e": DekGlyph(
        name="gab_vowel_e",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.3, 0.02)],
                is_shaded=False,
                thickness_factor=0.85,
            )
        ],
        delta_x=0.3,
        delta_y=0.02,
        description="Gabelsberger Vokal e: Enge Verbindung",
    ),

    # 'i': steiler Verbindungsstrich nach oben (Steil-/Hochstellung)
    "i": DekGlyph(
        name="gab_vowel_i",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.15, 0.45), (0.28, 0.85)],
                is_shaded=False,
                thickness_factor=0.9,
            )
        ],
        delta_x=0.28,
        delta_y=0.85,
        description="Gabelsberger Vokal i: Steile Hochstellung",
    ),

    # 'a': kräftig schattierte, mittellange Verbindung (Druckandeutung des 'a')
    "a": DekGlyph(
        name="gab_vowel_a",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.3, 0.05), (0.6, 0.0)],
                is_shaded=True,
                thickness_factor=2.5,
            )
        ],
        delta_x=0.6,
        delta_y=0.0,
        description="Gabelsberger Vokal a: Druckverstärkung (Schattierung)",
    ),

    # 'o': weit ausgerundeter Bogen (Rundwölbung für 'o')
    "o": DekGlyph(
        name="gab_vowel_o",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.25, -0.35), (0.55, -0.35), (0.75, -0.1)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.75,
        delta_y=-0.1,
        description="Gabelsberger Vokal o: Ausgerundete Verbindung",
    ),

    # 'u': tiefgezogene flache Verbindung (Tiefstellung für 'u')
    "u": DekGlyph(
        name="gab_vowel_u",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.35, -0.55), (0.7, -0.85)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.7,
        delta_y=-0.85,
        description="Gabelsberger Vokal u: Tiefe Senkung",
    ),

    # 'ei': längerer schräger Aufstrich mit leichter Druckschwellung
    "ei": DekGlyph(
        name="gab_vowel_ei",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.3, 0.4), (0.55, 0.75)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.55,
        delta_y=0.75,
        description="Gabelsberger Diphthong ei: Schräger Schwellaufstrich",
    ),

    # 'au': tiefer, kräftig schattierter Wölbbogen
    "au": DekGlyph(
        name="gab_vowel_au",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.35, -0.45), (0.75, -0.7)],
                is_shaded=True,
                thickness_factor=2.6,
            )
        ],
        delta_x=0.75,
        delta_y=-0.7,
        description="Gabelsberger Diphthong au: Schattierte Tiefstellung",
    ),
}


# =====================================================================
# 3. SIGEL (Kürzel des Gabelsberger-Systems)
# =====================================================================

GABELSBERGER_KUERZEL: Dict[str, DekGlyph] = {
    "und": DekGlyph(
        name="gab_sigel_und",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.18, -0.3), (0.38, -0.2), (0.48, 0.15)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.48,
        delta_y=0.15,
        description="Gabelsberger Sigel 'und'",
    ),

    "der": DekGlyph(
        name="gab_sigel_der",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.18, 0.15), (0.38, 0.05)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.38,
        delta_y=-0.75,
        description="Gabelsberger Sigel 'der'",
    ),

    "die": DekGlyph(
        name="gab_sigel_die",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.9), (0.15, 0.2), (0.3, 0.55), (0.45, 0.95)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.45,
        delta_y=0.05,
        description="Gabelsberger Sigel 'die'",
    ),

    "das": DekGlyph(
        name="gab_sigel_das",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.18, 0.1), (0.35, 0.35), (0.25, 0.5), (0.42, 0.1)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.42,
        delta_y=-0.7,
        description="Gabelsberger Sigel 'das'",
    ),

    "dem": DekGlyph(
        name="gab_sigel_dem",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.15, 0.1), (0.4, 0.45), (0.65, 0.0)],
                is_shaded=False,
                thickness_factor=1.1,
            )
        ],
        delta_x=0.65,
        delta_y=-0.8,
        description="Gabelsberger Sigel 'dem'",
    ),

    "den": DekGlyph(
        name="gab_sigel_den",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.15, 0.1), (0.35, 0.3), (0.55, 0.05)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.55,
        delta_y=-0.75,
        description="Gabelsberger Sigel 'den'",
    ),

    "nicht": DekGlyph(
        name="gab_sigel_nicht",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.2, 0.55), (0.38, 0.85)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.38,
        delta_y=0.85,
        description="Gabelsberger Sigel 'nicht'",
    ),

    "ist": DekGlyph(
        name="gab_sigel_ist",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.5), (0.15, 0.85), (0.05, 1.0), (-0.05, 0.8), (0.2, 0.5)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.2,
        delta_y=0.0,
        description="Gabelsberger Sigel 'ist'",
    ),

    "mit": DekGlyph(
        name="gab_sigel_mit",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.4), (0.2, 0.8), (0.4, 0.4), (0.6, 0.8), (0.8, 0.4)],
                is_shaded=False,
                thickness_factor=1.05,
            )
        ],
        delta_x=0.8,
        delta_y=0.0,
        description="Gabelsberger Sigel 'mit'",
    ),

    "für": DekGlyph(
        name="gab_sigel_fuer",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.2), (0.3, -0.5), (0.5, -0.3)],
                is_shaded=True,
                thickness_factor=2.0,
            )
        ],
        delta_x=0.5,
        delta_y=-1.5,
        description="Gabelsberger Sigel 'für'",
    ),

    "zu": DekGlyph(
        name="gab_sigel_zu",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.7), (0.2, -0.3), (0.35, -0.1)],
                is_shaded=True,
                thickness_factor=2.0,
            )
        ],
        delta_x=0.35,
        delta_y=-0.8,
        description="Gabelsberger Sigel 'zu'",
    ),

    "werden": DekGlyph(
        name="gab_sigel_werden",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.85), (0.25, 0.6), (0.3, 0.1), (0.45, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.45,
        delta_y=-0.85,
        description="Gabelsberger Sigel 'werden'",
    ),
}

