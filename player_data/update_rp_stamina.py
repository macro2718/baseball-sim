import json
import os
import random


def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "players.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for p in data.get("pitchers", []):
        name = p.get("name", "")
        ptype = p.get("pitcher_type", "")
        if ptype == "SP" and "('24)" not in name:
            # Uniform integer in [65, 95]
            p["stamina"] = int(random.randint(70, 100))
            count += 1

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated stamina for {count} relief pitchers (excluding ('24)).")


if __name__ == "__main__":
    main()

