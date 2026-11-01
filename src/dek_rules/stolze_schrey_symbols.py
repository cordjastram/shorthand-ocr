"""
Stolze-Schrey-Kurzschrift (Vereinigtes System Stolze-Schrey, 1897) - Symboldefinitionen.

Charakteristika des Systems Stolze-Schrey (Wilhelm Stolze / Ferdinand Schrey, 1897):
- Klar strukturierte, geometrisch-strenge und gleichmäßig geneigte Abstrich-Systematik
- Alle Hauptkonsonanten sind Abstriche (von rechts oben nach links/rechts unten),
  verbunden durch Haarstrich-Aufstriche
- Sehr systematische Dreistufigkeit und Vokalisation durch Kombination aus:
  * Weite der Verbindung (eng vs. weit)
  * Höhe zum Vorzeichen (Grundstellung, Hochstellung, Tiefstellung)
  * Druckstärke (unschattiert vs. schattiert am nachfolgenden Konsonanten)
    - e: eng, gleiche Höhe, unschattiert
    - a: weit, gleiche Höhe, schattiert
    - o: weit, gleiche Höhe, unschattiert
    - i: eng, hochgestellt, unschattiert
    - u: weit, tiefgestellt, unschattiert
    - ei: weit, schräger Aufstrich
    - au: weit, tiefgestellt, schattiert
"""

from typing import Dict
from src.dek_rules.symbols import DekGlyph, StrokeSegment


# =====================================================================
# 1. KONSONANTEN (Grundzeichen nach Stolze-Schrey, 1897)
# =====================================================================

STOLZE_SCHREY_CONSONANTS: Dict[str, DekGlyph] = {
    # 't': zweistufiger gerader Abstrich (unschattiert)
    "t": DekGlyph(
        name="ss_t",
        segments=[
            StrokeSegment(
                points=[(0.0, 2.0), (0.55, 0.0)],
                is_shaded=False,
                thickness_factor=1.05,
            )
        ],
        delta_x=0.55,
        delta_y=-2.0,
        start_level=2.0,
        end_level=0.0,
        description="Stolze-Schrey t: Zweistufiger gerader Abstrich",
    ),

    # 'd': einstufiger gerader Abstrich (unschattiert)
    "d": DekGlyph(
        name="ss_d",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.3, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.3,
        delta_y=-1.0,
        start_level=1.0,
        end_level=0.0,
        description="Stolze-Schrey d: Einstufiger gerader Abstrich",
    ),

    # 'b': einstufiger Abstrich mit Fußrundung nach rechts
    "b": DekGlyph(
        name="ss_b",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.2, 0.25), (0.35, 0.0), (0.5, 0.1)],
                is_shaded=False,
                thickness_factor=1.1,
            )
        ],
        delta_x=0.5,
        delta_y=-0.9,
        start_level=1.0,
        end_level=0.1,
        description="Stolze-Schrey b: Einstufiger Fußbogen-Abstrich",
    ),

    # 'p': zweistufiger Abstrich mit kräftiger Schattierung und Fußbogen
    "p": DekGlyph(
        name="ss_p",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.8), (0.35, 0.3), (0.5, 0.0), (0.65, 0.15)],
                is_shaded=True,
                thickness_factor=2.5,
            )
        ],
        delta_x=0.65,
        delta_y=-1.65,
        start_level=1.8,
        end_level=0.15,
        description="Stolze-Schrey p: Zweistufiger schattierter Abstrich",
    ),

    # 'w': einstufiger, sanft nach rechts gewölbter Abstrich
    "w": DekGlyph(
        name="ss_w",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.22, 0.65), (0.3, 0.25), (0.22, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.22,
        delta_y=-1.0,
        start_level=1.0,
        end_level=0.0,
        description="Stolze-Schrey w: Rechtsgewölbter Einstufen-Abstrich",
    ),

    # 'f': zweistufiger, leicht geschwungener Abstrich mit Schattierung
    "f": DekGlyph(
        name="ss_f",
        segments=[
            StrokeSegment(
                points=[(0.0, 2.0), (0.25, 1.0), (0.5, 0.0)],
                is_shaded=True,
                thickness_factor=2.3,
            )
        ],
        delta_x=0.5,
        delta_y=-2.0,
        start_level=2.0,
        end_level=0.0,
        description="Stolze-Schrey f: Zweistufiger Druckabstrich",
    ),

    # 'm': einstufiger Kopfbogen-Abstrich (mit leichtem Druck)
    "m": DekGlyph(
        name="ss_m",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.75), (0.2, 1.0), (0.4, 0.75), (0.55, 0.0)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.55,
        delta_y=-0.75,
        start_level=0.75,
        end_level=0.0,
        description="Stolze-Schrey m: Schattierter Kopfbogen-Abstrich",
    ),

    # 'n': halbstufiger kleiner Wellen-/Muldenabstrich
    "n": DekGlyph(
        name="ss_n",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.5), (0.15, 0.1), (0.32, 0.0), (0.45, 0.2)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.45,
        delta_y=-0.3,
        start_level=0.5,
        end_level=0.2,
        description="Stolze-Schrey n: Halbstufiger Muldenstrich",
    ),

    # 'l': aufwärts gerichtete kleine Schleife (1 Stufe)
    "l": DekGlyph(
        name="ss_l",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.25, 0.9), (0.15, 1.15), (0.02, 0.9), (0.2, 0.2), (0.38, 0.0)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.38,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Stolze-Schrey l: Einstufige Oberschleife",
    ),

    # 'r': kleiner Kreisringel auf der Linie
    "r": DekGlyph(
        name="ss_r",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.14, 0.28), (0.04, 0.42), (-0.08, 0.28), (0.06, 0.05), (0.22, 0.0)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.22,
        delta_y=0.0,
        start_level=0.0,
        end_level=0.0,
        description="Stolze-Schrey r: Kleiner Kreisringel",
    ),

    # 's': halbstufiger feiner Abstrich mit kleinem Haken
    "s": DekGlyph(
        name="ss_s",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.6), (0.18, 0.0), (0.28, 0.12)],
                is_shaded=False,
                thickness_factor=0.95,
            )
        ],
        delta_x=0.28,
        delta_y=-0.48,
        start_level=0.6,
        end_level=0.12,
        description="Stolze-Schrey s: Halbstufiger Hakenabstrich",
    ),

    # 'sch': zweistufiger Abstrich mit deutlicher Fußschlinge
    "sch": DekGlyph(
        name="ss_sch",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.6), (0.35, -0.2), (0.2, -0.55), (0.05, -0.3), (0.35, 0.0), (0.55, 0.1)],
                is_shaded=False,
                thickness_factor=1.1,
            )
        ],
        delta_x=0.55,
        delta_y=-1.5,
        start_level=1.6,
        end_level=0.1,
        description="Stolze-Schrey sch: Langer Abstrich mit Fußschlinge",
    ),

    # 'ch': einstufiger S-förmig geschwungener Abstrich
    "ch": DekGlyph(
        name="ss_ch",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.1), (-0.08, 0.75), (0.22, 0.35), (0.35, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.35,
        delta_y=-1.1,
        start_level=1.1,
        end_level=0.0,
        description="Stolze-Schrey ch: Geschwungener S-Abstrich",
    ),

    # 'g': einstufiger, nach links offener Bogenabstrich mit Unterlänge
    "g": DekGlyph(
        name="ss_g",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.6), (0.25, -0.3), (0.3, -0.9), (0.1, -1.05), (0.0, -0.7), (0.35, 0.0)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.35,
        delta_y=-0.6,
        start_level=0.6,
        end_level=0.0,
        description="Stolze-Schrey g: Schattierte Unterlängenschleife",
    ),

    # 'k': einstufiger Kopfbogen-Abstrich (unschattiert)
    "k": DekGlyph(
        name="ss_k",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.7), (0.18, 0.95), (0.35, 0.65), (0.48, 0.0)],
                is_shaded=False,
                thickness_factor=1.05,
            )
        ],
        delta_x=0.48,
        delta_y=-0.7,
        start_level=0.7,
        end_level=0.0,
        description="Stolze-Schrey k: Einstufiger Kopfbogen",
    ),

    # 'z': einstufiger, scharf geknickter Abstrich mit Druck
    "z": DekGlyph(
        name="ss_z",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.1), (0.25, 0.5), (0.1, 0.4), (0.38, 0.0)],
                is_shaded=True,
                thickness_factor=2.2,
            )
        ],
        delta_x=0.38,
        delta_y=-1.1,
        start_level=1.1,
        end_level=0.0,
        description="Stolze-Schrey z: Geknickter Druckabstrich",
    ),
}


# =====================================================================
# 2. VOKAL-VERBINDUNGEN (Systematische Aufstrich-Verbindungen)
# Da Konsonanten in Stolze-Schrey fast immer Abstriche sind, führen
# die Vokalverbindungen als Aufstriche zum Kopf des nächsten Zeichens!
# =====================================================================

STOLZE_SCHREY_VOWELS: Dict[str, DekGlyph] = {
    # 'e': enger, gerader Haarstrich-Aufstrich auf Normalhöhe
    "e": DekGlyph(
        name="ss_vowel_e",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.32, 0.75)],
                is_shaded=False,
                thickness_factor=0.85,
            )
        ],
        delta_x=0.32,
        delta_y=0.75,
        description="Stolze-Schrey Vokal e: Enger Aufstrich",
    ),

    # 'i': enger, steiler Aufstrich in die Hochstellung (+1.35 Stufen)
    "i": DekGlyph(
        name="ss_vowel_i",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.28, 1.35)],
                is_shaded=False,
                thickness_factor=0.85,
            )
        ],
        delta_x=0.28,
        delta_y=1.35,
        description="Stolze-Schrey Vokal i: Enger Steilaufstrich (Hochstellung)",
    ),

    # 'a': weiter Aufstrich mit kräftiger Druckverstärkung am Übergang
    "a": DekGlyph(
        name="ss_vowel_a",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.4), (0.75, 0.85)],
                is_shaded=True,
                thickness_factor=2.3,
            )
        ],
        delta_x=0.75,
        delta_y=0.85,
        description="Stolze-Schrey Vokal a: Weite Verbindung mit Druck",
    ),

    # 'o': weiter, flach geschwungener Aufstrich ohne Druck
    "o": DekGlyph(
        name="ss_vowel_o",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, 0.25), (0.8, 0.75)],
                is_shaded=False,
                thickness_factor=0.9,
            )
        ],
        delta_x=0.8,
        delta_y=0.75,
        description="Stolze-Schrey Vokal o: Weiter unverdickter Aufstrich",
    ),

    # 'u': weiter, flach bleibender Verbindungsstrich (Tiefstellung)
    "u": DekGlyph(
        name="ss_vowel_u",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.4, -0.1), (0.8, 0.2)],
                is_shaded=False,
                thickness_factor=0.9,
            )
        ],
        delta_x=0.8,
        delta_y=0.2,
        description="Stolze-Schrey Vokal u: Weite Tiefstellung",
    ),

    # 'ei': weiter, gestreckter Schrägaufstrich
    "ei": DekGlyph(
        name="ss_vowel_ei",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.45, 0.6), (0.85, 1.25)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.85,
        delta_y=1.25,
        description="Stolze-Schrey Diphthong ei: Weiter Schrägaufstrich",
    ),

    # 'au': weiter, tief gehaltener Strich mit Druckverstärkung
    "au": DekGlyph(
        name="ss_vowel_au",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.0), (0.45, -0.15), (0.85, 0.25)],
                is_shaded=True,
                thickness_factor=2.4,
            )
        ],
        delta_x=0.85,
        delta_y=0.25,
        description="Stolze-Schrey Diphthong au: Weite Tiefstellung mit Druck",
    ),
}


# =====================================================================
# 3. KÜRZEL / SIGEL (Stolze-Schrey)
# =====================================================================

STOLZE_SCHREY_KUERZEL: Dict[str, DekGlyph] = {
    "und": DekGlyph(
        name="ss_kuerzel_und",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.1), (0.15, -0.15), (0.35, 0.25)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.35,
        delta_y=0.15,
        description="Stolze-Schrey Kürzel 'und'",
    ),

    "der": DekGlyph(
        name="ss_kuerzel_der",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.3, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.3,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'der'",
    ),

    "die": DekGlyph(
        name="ss_kuerzel_die",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.5), (0.3, 0.5)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.3,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'die' (hochgestellt)",
    ),

    "das": DekGlyph(
        name="ss_kuerzel_das",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.3, 0.0)],
                is_shaded=True,
                thickness_factor=2.3,
            )
        ],
        delta_x=0.3,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'das' (schattiertes d)",
    ),

    "dem": DekGlyph(
        name="ss_kuerzel_dem",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.25, 0.2), (0.45, 0.5), (0.65, 0.0)],
                is_shaded=True,
                thickness_factor=2.0,
            )
        ],
        delta_x=0.65,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'dem'",
    ),

    "den": DekGlyph(
        name="ss_kuerzel_den",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.25, 0.1), (0.45, 0.0), (0.6, 0.2)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.6,
        delta_y=-0.8,
        description="Stolze-Schrey Kürzel 'den'",
    ),

    "nicht": DekGlyph(
        name="ss_kuerzel_nicht",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.9), (0.15, 0.5), (0.35, 0.7)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.35,
        delta_y=-0.2,
        description="Stolze-Schrey Kürzel 'nicht'",
    ),

    "ist": DekGlyph(
        name="ss_kuerzel_ist",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.9), (0.18, 0.35), (0.28, 0.45)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.28,
        delta_y=-0.45,
        description="Stolze-Schrey Kürzel 'ist'",
    ),

    "mit": DekGlyph(
        name="ss_kuerzel_mit",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.2), (0.2, 1.4), (0.4, 1.15), (0.55, 0.4)],
                is_shaded=True,
                thickness_factor=2.1,
            )
        ],
        delta_x=0.55,
        delta_y=-0.8,
        description="Stolze-Schrey Kürzel 'mit'",
    ),

    "für": DekGlyph(
        name="ss_kuerzel_fuer",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.5), (0.4, -0.5)],
                is_shaded=True,
                thickness_factor=2.2,
            )
        ],
        delta_x=0.4,
        delta_y=-2.0,
        description="Stolze-Schrey Kürzel 'für'",
    ),

    "zu": DekGlyph(
        name="ss_kuerzel_zu",
        segments=[
            StrokeSegment(
                points=[(0.0, 0.8), (0.25, -0.2)],
                is_shaded=True,
                thickness_factor=2.2,
            )
        ],
        delta_x=0.25,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'zu'",
    ),

    "werden": DekGlyph(
        name="ss_kuerzel_werden",
        segments=[
            StrokeSegment(
                points=[(0.0, 1.0), (0.22, 0.65), (0.3, 0.25), (0.22, 0.0)],
                is_shaded=False,
                thickness_factor=1.0,
            )
        ],
        delta_x=0.22,
        delta_y=-1.0,
        description="Stolze-Schrey Kürzel 'werden'",
    ),
}

