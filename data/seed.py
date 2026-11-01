"""Seed the expense database with sample rows so the personal-data tool has
something to answer with on first run.  Run:  python data/seed.py"""
import random, sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from tools.personal_data import PersonalDataTool

CATALOG = {
    "food":      [("Lunch", 180, 520), ("Groceries", 800, 2400), ("Coffee", 120, 320),
                  ("Dinner out", 600, 1800)],
    "transport": [("Cab", 150, 480), ("Metro card", 200, 600), ("Fuel", 900, 2200)],
    "utilities": [("Electricity", 900, 2200), ("Internet", 799, 1299), ("Phone", 299, 799)],
    "shopping":  [("Clothes", 800, 3500), ("Headphones", 1500, 6000), ("Books", 300, 1200)],
    "health":    [("Pharmacy", 200, 900), ("Gym", 1200, 2500)],
    "entertainment": [("Movie", 300, 800), ("Subscription", 149, 799)],
}


def main() -> int:
    cfg = load_config()
    tool = PersonalDataTool(cfg.tools, cfg.db_path())
    if tool.count() > 0:
        print(f"Database already has {tool.count()} rows. Nothing seeded.")
        return 0

    random.seed(7)
    today = date.today()
    n = 0
    for days_ago in range(0, 75):
        day = today - timedelta(days=days_ago)
        for _ in range(random.choice([0, 1, 1, 2, 2, 3])):
            category = random.choice(list(CATALOG))
            desc, lo, hi = random.choice(CATALOG[category])
            tool.add_expense(day.isoformat(), category, desc,
                             round(random.uniform(lo, hi), 2))
            n += 1
    print(f"Seeded {n} expenses into {cfg.db_path()}")
    print("Try asking: 'How much did I spend this month?'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
