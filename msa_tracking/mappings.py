"""Hand-edited standardization dictionaries for the MSA tracking notebooks.

When the run log warns about an unmapped value, add it here.
"""

# Approval level as written by the manufacturer -> standardized level.
# Tesla and Enphase only publish lists of approved utilities, so being on the
# list is recorded as "Listed". Anything missing here becomes "UNMAPPED".
APPROVAL_STD = {
    "Listed": "approved",
    "Approved": "approved",
    "Pilot in progress": "pilot",
    "Approvals on case-by-case basis": "case_by_case",
    "In Progress (TBD)": "in_progress",
    "Expected Q, YYYY": "expected",
    "N/A": "not_applicable",
}

# Enphase status pills that describe a limitation rather than an approval
# level. These go in `notes` and the approval stays "Listed".
ENPHASE_LIMITATION_PILLS = {"Ring-type meter base only"}

# Standardized level -> status shown in the per-brand summary. A brand's status
# for a utility is its best one across its products, in SUMMARY_ORDER. Not
# approved (N/A, or not listed at all) is blank so approvals stand out.
BRANDS = ["Tesla", "Enphase", "ConnectDER"]
APPROVED = "Approved"
SUMMARY_STATUS = {
    "approved": APPROVED,
    "pilot": "Pilot",
    "case_by_case": "Case-by-case",
    "in_progress": "Pending",
    "expected": "Pending",
    "not_applicable": "",
}
SUMMARY_ORDER = [APPROVED, "Pilot", "Case-by-case", "Pending", ""]

# Standardized utility name -> the spellings manufacturers use for it. Any name
# not listed passes through unchanged. If you rename a standardized name, rename
# it in data/msa_eia_crosswalk.csv too.
UTILITY_VARIANTS = {
    "Arizona Public Service": ["Arizona Public Service (APS)", "Arizona Public Service Company"],
    "Atlantic City Electric": ["ACE", "Atlantic City Electric (ACE)"],
    "Austin Energy": ["Austin Energy (City of Austin)"],
    "Baltimore Gas and Electric": ["Baltimore Gas & Electric", "Baltimore Gas & Electric (BGE)",
                                   "Baltimore Gas and Electric (BGE)"],
    "Black Hills Energy": ["Black Hills Corporation"],
    "Bluebonnet Electric Cooperative": ["Bluebonnet Electric Coop (BB)"],
    "Buckeye Rural Electric Cooperative": ["Buckeye Rural Electric", "Buckeye Rural Electrical Cooperative"],
    "City of Tallahassee": ["City of Tallahassee Electric Utility"],
    "Commonwealth Edison": ["ComEd"],
    "CoServ Electric Cooperative": ["CoServ"],
    "Delmarva Power": ["Delmarva", "DPL"],
    # Enphase's "Eau Clair Utility" is taken to be the co-op; the city of Eau
    # Claire has no municipal electric utility.
    "Eau Claire Energy Cooperative": ["Eau Clair Utility", "Eau Claire Energy Coop"],
    "Fort Collins Light and Power": ["City of Ft. Collins Utility", "Fort Collins Light & Power",
                                     "Fort Collins Light and Power (FCLP)"],
    "Grayson-Collin Electric Cooperative": ["Grayson-Collin Electric Cooperative (GCEC)"],
    "Green Mountain Power": ["Green Mountain Power (GMP)"],
    # Hawai'i Island (HELCO) and Maui Electric operate under the single
    # Hawaiian Electric brand; Tesla and Enphase list only the brand.
    "Hawaiian Electric": ["Hawaiian Electric Company", "Heco", "Hawai'i Electric Light Co", "Maui Electric"],
    "Hohokam Irrigation and Drainage District": ["Hohokam Irrigation & Drainage District",
                                                 "Hohokam Irrigation and Power"],
    "Jersey Central Power and Light": ["Jersey Central Power & Light"],
    "La Plata Electric Association": ["La Plata Electric Association (LPEA)"],
    "Los Alamos County Utilities": ["Los Alamos Department of public utilities"],
    "NV Energy": ["NV Energy (South and North)"],
    "Omaha Public Power District": ["Omaha Public Power District (OPPD)"],
    "Orlando Utilities Commission": ["OUC (Orlando Utilities Commission)"],
    "Pacific Gas and Electric": ["Pacific Gas & Electric", "Pacific Gas and Electric Company",
                                 "Pacific Gas and Electric Company (PG&E)"],
    "Pepco": ["PEPCO", "Potomac Electric Power Company (Pepco)"],
    "Plumas-Sierra Rural Electric Cooperative": ["Plumas-Sierra Rural Electric co-op"],
    "Poudre Valley Rural Electric Association": ["Poudre Valley Electric Association"],
    "Public Service Electric and Gas": ["Public Service Electirc & Gas (PSE&G) NJ",
                                        "Public Service Electric & Gas Company"],
    "Rocky Mountain Power": ["Rocky Mountain Power (RMP)"],
    "Sacramento Municipal Utility District": ["Sacramento Municipal Utility District (SMUD)"],
    "Salt River Project": ["Salt River Project (SRP)"],
    "San Diego Gas and Electric": ["San Diego Gas & Electric", "San Diego Gas & Electric (SDG&E)"],
    "Saskatoon Light & Power": ["Saskatoon Light & Power (SL&P)"],
    "Southern California Edison": ["Southern California Edison (SCE)"],
    "Southern Maryland Electric Cooperative": ["Southern Maryland Electric Coop (SMECO)"],
    "Sulphur Springs Valley Electric Cooperative": ["Sulphur Spring Valley Electric",
                                                    "Sulphur Springs Valley Electric Coop"],
    "Trico Electric Cooperative": ["Trico electric co-op"],
    "Tucson Electric Power": ["Tucson Electric Power (TEP)"],
    "Vermont Electric Cooperative": ["Vermont Electric Coop"],
    "Washington Electric Cooperative": ["Washington Electric Coop", "Washington Electrical Cooperative"],
    "West River Electric Association": ["West River Electric Association (WREA)"],
    "Westerville Electric": ["City of Westerville OH"],
    "Xcel Energy": ["Xcel Energy-Colorado"],
    "Yampa Valley Electric Association": ["Yampa Valley Electrical Association (YVEA)"],
    "Yellow Springs Electric": ["Village of Yellow Springs"],
}
UTILITY_STD = {variant: std for std, variants in UTILITY_VARIANTS.items() for variant in variants}

# State or province name as written -> two-letter code. STATE_NAMES maps back,
# using the first name listed for each code.
STATE_STD = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Washington DC": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR",
    # Canada (Enphase lists a few provinces)
    "Nova Scotia": "NS", "Saskatchewan": "SK", "Alberta": "AB", "British Columbia": "BC",
    "Ontario": "ON", "Quebec": "QC", "Manitoba": "MB", "New Brunswick": "NB",
}
STATE_NAMES = {}
for _name, _code in STATE_STD.items():
    STATE_NAMES.setdefault(_code, _name)
