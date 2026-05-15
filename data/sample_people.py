import numpy as np
import random
from pathlib import Path


# Resolved relative to this module so the import works regardless of cwd.
FIELDS_DIR = Path(__file__).parent / "fields"

def load_lines(filename):
    with open(FIELDS_DIR / filename) as f:
        return [line.strip() for line in f if line.strip()]

first_names  = load_lines("first_name.txt")    # 400
middle_names = load_lines("middle_name.txt")   # 400
last_names   = load_lines("last_name.txt")     # 1000
cities       = load_lines("city.txt")          # 200  (e.g. "New York City, NY")
universities = load_lines("university.txt")    # 300
fields       = load_lines("field.txt")         # 100
# jobs     # capo doesnt use jobs.

# company.txt is "Name; City, ST" — split into parallel lists
company_names  = []
company_cities = []
for line in load_lines("company.txt"):
    name, city = line.split(";", 1)
    company_names.append(name.strip())
    company_cities.append(city.strip())
    

MONTHS = ["January", "February", "March", "April", "May", "June","July", "August", "September", "October", "November", "December"]

def sample_people(N, seed=0):
    rng = random.Random(seed)
    n_first, n_mid, n_last = len(first_names), len(middle_names), len(last_names)
    name_space = n_first * n_mid * n_last
    name_indices = rng.sample(range(name_space), N)

    people = []
    for raw_idx, idx in enumerate(name_indices):
        i = idx // (n_mid * n_last)
        j = (idx // n_last) % n_mid
        k = idx % n_last

        # Gender bit comes from the FIRST name's slot (0..199 male, 200..399 female).
        # Force the middle name into the matching half so the bio is gender-coherent.
        is_female = i >= 200 #Figure out if is female
        if is_female and j < 200:
            j = 200 + (j % 200)            # bump male middle into female half
        elif (not is_female) and j >= 200:
            j = j % 200                    # bump female middle into male half

        # Make `id` parity match gender so get_text_simple3's `id%2==0 -> He` works:
        #   even id -> 'He' -> male; odd id -> 'She' -> female
        pid = 2 * raw_idx + (1 if is_female else 0)

        c = rng.randrange(len(company_names))
        people.append({
            "id": pid,
            "first_name":  first_names[i],
            "middle_name": middle_names[j],
            "last_name":   last_names[k],
            "birthday":    rng.randint(1, 28),
            "birthmonth":  MONTHS[rng.randrange(12)],
            "birthyear":   rng.randrange(1700, 1900),
            "birthcity":   rng.choice(cities),
            "university":  rng.choice(universities),
            "field":       rng.choice(fields),
            "company1name": company_names[c],
            "company1city": company_cities[c],
        })
    return people
