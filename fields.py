"""
Field schema + controlled vocabularies for the Aude/Gironde Military Registers project.
Mirrors the 'Military' doctype in the CI spec (62353.IDX.001) and the .db3 Record schema.
Each field: (Label as shown to keyers, internal Ancestry field name, dictionary/vocab key).
"""

# (label, internal_name, vocab)  -- vocab is None for free-text
MILITARY_FIELDS = [
    ("Prefix",               "SelfNamePrefix",                "prefix"),
    ("Given Name",           "SelfGivenName",                 None),
    ("Surname",              "SelfSurname",                   None),
    ("Suffix",               "SelfNameSuffix",                "suffix"),
    ("Birth Day",            "SelfBirthDay",                  "day"),
    ("Birth Month",          "SelfBirthMonth",                "month"),
    ("Birth Year",           "SelfBirthYear",                 "year"),
    ("Birth Commune",        "SelfBirthCity",                 "city"),
    ("Birth Canton",         "SelfBirthCounty",               "city"),
    ("Birth Departement",    "SelfBirthState",                "state"),
    ("Hair Color",           "SelfMilitaryHairColor",         "hair"),
    ("Eye Color",            "SelfMilitaryEyeColor",          "eye"),
    ("Height",               "SelfMilitaryHeight",            None),
    ("Father Prefix",        "FatherNamePrefix",              "prefix"),
    ("Father Given Name",    "FatherGivenName",               None),
    ("Father Surname",       "FatherSurname",                 None),
    ("Father Suffix",        "FatherNameSuffix",              "suffix"),
    ("Deceased Father",      "FatherIsDeceased",              "yn"),
    ("Mother Prefix",        "MotherNamePrefix",              "prefix"),
    ("Mother Given Name",    "MotherGivenName",               None),
    ("Mother Maiden Name",   "MotherMaidenName",              None),
    ("Mother Surname",       "MotherSurname",                 None),
    ("Mother Suffix",        "MotherNameSuffix",              "suffix"),
    ("Deceased Mother",      "MotherIsDeceased",              "yn"),
    ("Domicile",             "SelfResidencePlace",            None),
    ("Residence Commune",    "SelfResidenceCity",             "city"),
    ("Residence Canton",     "SelfResidenceCounty",           "city"),
    ("Enlistment City",      "SelfMilitaryEnlistmentCity",    "city"),
    ("Enlistment Departement","SelfMilitaryEnlistmentState",  "state"),
    ("Regiment",             "SelfMilitaryRegiment",          None),
    ("Unit",                 "SelfMilitaryMilitaryUnit",      None),
    ("Branch",               "SelfMilitaryServiceBranch",     None),
    ("Compagnie",            "SelfMilitaryMilitaryCompany",   None),
    ("Battalion",            "Notes",                         None),
    ("Rank",                 "SelfMilitaryRank",              None),
    ("Discharge Day",        "SelfMilitaryDischargeDay",      "day"),
    ("Discharge Month",      "SelfMilitaryDischargeMonth",    "month"),
    ("Discharge Year",       "SelfMilitaryDischargeYear",     "year"),
    ("Death Day",            "SelfDeathDay",                  "day"),
    ("Death Month",          "SelfDeathMonth",                "month"),
    ("Death Year",           "SelfDeathYear",                 "year"),
    ("Death Commune",        "SelfDeathCity",                 "city"),
    ("Occupation",           "SelfOccupation",                None),
    ("Classe Year",          "SelfMilitaryEnlistmentYear",    "year"),
    ("Entry Number",         "LineNumber",                    None),
    ("Event Type",           "RecordType",                    "event"),
]

FIELD_LABELS = [f[0] for f in MILITARY_FIELDS]
LABEL_TO_KEY = {f[0]: f[1] for f in MILITARY_FIELDS}
LABEL_TO_VOCAB = {f[0]: f[2] for f in MILITARY_FIELDS}

# ---------------------------------------------------------------------------
# Controlled vocabularies (closed sets get fuzzy-snapped during post-correction)
# ---------------------------------------------------------------------------

# Month -> canonical lowercase French name (matches the actual keyed
# reference convention — NOT an English 3-letter code, which real keyers
# don't use). Includes French/English OCR-variant spellings, the archaic
# "7bre/8bre/9bre/Xbre" abbreviations (Sept/Oct/Nov/Dec, from Latin septem/
# octo/novem/decem), and bare month numbers, all snapping to the French name.
MONTHS = {
    "janvier": ["janvier", "janv", "jan", "january", "janvr", "jean", "1"],
    "février": ["fevrier", "février", "fev", "fév", "feb", "february", "fevr", "2"],
    "mars": ["mars", "mar", "march", "3"],
    "avril": ["avril", "avr", "apr", "april", "4"],
    "mai": ["mai", "may", "5"],
    "juin": ["juin", "jun", "june", "6"],
    "juillet": ["juillet", "juil", "jul", "july", "7"],
    "août": ["aout", "août", "aug", "august", "8"],
    "septembre": ["septembre", "sep", "sept", "september", "7bre", "9"],
    "octobre": ["octobre", "oct", "october", "8bre", "10"],
    "novembre": ["novembre", "nov", "november", "9bre", "11"],
    "décembre": ["decembre", "décembre", "dec", "december", "10bre", "xbre", "12"],
    # French Republican months (kept as-is per CI, mapped to themselves)
    "Vendemiaire": ["vendemiaire", "vendémiaire"],
    "Brumaire": ["brumaire"], "Frimaire": ["frimaire"], "Nivose": ["nivose", "nivôse"],
    "Pluviose": ["pluviose", "pluviôse"], "Ventose": ["ventose", "ventôse"],
    "Germinal": ["germinal"], "Floreal": ["floreal", "floréal"], "Prairial": ["prairial"],
    "Messidor": ["messidor"], "Thermidor": ["thermidor"], "Fructidor": ["fructidor"],
}

HAIR = {  # CI abbreviations
    "BR": ["brown", "brun", "bruns", "brunet", "marron", "chatain fonce"],
    "BK": ["black", "noir", "noirs"],
    "LT": ["light", "clair", "clairs"],
    "FR": ["fair", "blond", "blonds"],
    "RD": ["red", "roux"],
    "CH": ["chatain", "châtain", "chatains", "châtains"],
    "GY": ["grey", "gray", "gris"],
    "WH": ["white", "blanc", "blancs"],
}

EYE = {
    "BL": ["blue", "bleu", "bleus"],
    "BR": ["brown", "brun", "bruns", "marron"],
    "GR": ["green", "vert", "verts"],
    "GB": ["grey blue", "gris bleu", "gris-bleu"],
    "GY": ["grey", "gray", "gris"],
    "BG": ["blue green", "bleu vert", "bleu-vert"],
    "H":  ["hazel", "noisette"],
    "CH": ["chatain", "châtain"],
}

# Départements likely to appear (Aude project + Gironde source archive + neighbours)
STATES = [
    "Aude", "Tarn", "Gironde", "Gard", "Herault", "Ariege", "Pyrenees-Orientales",
    "Haute-Garonne", "Tarn-et-Garonne", "Aveyron", "Lozere", "Dordogne",
    "Lot", "Lot-et-Garonne", "Landes", "Charente", "Charente-Maritime",
    "Bouches-du-Rhone", "Correze",
]

# Seed commune list (Aude + Gironde). Not exhaustive; used for fuzzy hints only.
CITIES = [
    # Aude
    "Carcassonne", "Narbonne", "Castelnaudary", "Limoux", "Lezignan-Corbieres",
    "Quillan", "Gimel", "Coursan", "Trebes", "Bram", "Capendu", "Conques-sur-Orbiel",
    "Fanjeaux", "Chalabre", "Belpech", "Alzonne", "Montreal", "Salles-sur-l'Hers",
    # Gironde
    "Bordeaux", "Libourne", "Blaye", "Bazas", "La Reole", "Langon", "Arcachon",
    "Lesparre-Medoc", "Podensac", "Cadillac", "Saint-Andre-de-Cubzac", "Pauillac",
    "Pessac", "Talence", "Merignac", "Begles", "Villenave-d'Ornon",
]

PREFIX = ["Sieur", "Monsieur", "M.", "Madame", "Mademoiselle", "Mme", "Mlle", "Dame"]
SUFFIX = ["Jr", "Sr", "II", "III", "fils", "aine", "cadet", "jeune"]
EVENT = {"Military": ["military", "militaire"], "Coverpage": ["coverpage", "cover", "couverture"]}
YN = {"Y": ["y", "yes", "oui", "feu", "feue", "defunt", "defunte", "decede"]}

VOCAB_SETS = {  # vocab name -> {canonical: [variants]}  OR  [list of canonicals]
    "month": MONTHS, "hair": HAIR, "eye": EYE, "state": STATES,
    "city": CITIES, "prefix": PREFIX, "suffix": SUFFIX, "event": EVENT, "yn": YN,
}
