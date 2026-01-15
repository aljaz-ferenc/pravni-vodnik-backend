from typing import TypedDict


class TestQuery(TypedDict):
    query: str
    law: str
    threshold: float
    max_queries: int


test_queries: list[TestQuery] = [
    # GDPR
    {
        "query": "Koliko časa sme spletna trgovina hraniti osebne podatke kupca",
        "law": "gdpr",
        "threshold": 0.92,  # high, narrow focus
        "max_queries": 2,
    },
    {
        "query": "Pravila o hranjenju osebnih podatkov v elektronski trgovini po slovenski zakonodaji",
        "law": "gdpr",
        "threshold": 0.92,
        "max_queries": 2,
    },
    {
        "query": "Katere pravice ima kupec glede osebnih podatkov po GDPR",
        "law": "gdpr",
        "threshold": 0.88,  # broad user intent
        "max_queries": 3,
    },
    # KZ-1 (Kazenski zakonik)
    {
        "query": "Kakšne kazni so predpisane za povzročitev telesne poškodbe",
        "law": "kz-1",
        "threshold": 0.90,
        "max_queries": 2,
    },
    {
        "query": "Kaj določa KZ-1 o kazenski odgovornosti mladoletnikov",
        "law": "kz-1",
        "threshold": 0.88,
        "max_queries": 2,
    },
    {
        "query": "Katere so določbe KZ-1 o gospodarskem kriminalu",
        "law": "kz-1",
        "threshold": 0.85,
        "max_queries": 3,
    },
    # ZKP (Zakon o kazenskem postopku)
    {
        "query": "Kako poteka postopek preiskave po ZKP",
        "law": "zkp",
        "threshold": 0.90,
        "max_queries": 2,
    },
    {
        "query": "Pravice osumljenca med kazenskim postopkom po ZKP",
        "law": "zkp",
        "threshold": 0.88,
        "max_queries": 3,
    },
    {
        "query": "Posebni postopki po ZKP, vključno z mladoletniki in pravnimi osebami",
        "law": "zkp",
        "threshold": 0.85,
        "max_queries": 3,
    },
    # Ustava (Constitution)
    {
        "query": "Katere človekove pravice so zagotovljene v Ustavi RS",
        "law": "ustava",
        "threshold": 0.88,
        "max_queries": 3,
    },
    {
        "query": "Kaj določa Ustava RS o svobodi govora in tisku",
        "law": "ustava",
        "threshold": 0.90,
        "max_queries": 2,
    },
    {
        "query": "Pravice posameznika do zasebnosti po Ustavi RS",
        "law": "ustava",
        "threshold": 0.88,
        "max_queries": 2,
    },
]
