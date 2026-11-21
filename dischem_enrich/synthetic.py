"""Seeded, independently phrased toy catalogue; structured names are ASSUMED."""

import json
from pathlib import Path
import random

from .data import write_csv
from .schema import Schema
from .pipeline import write_submission

ASSUMED_FIELDS = {
    "article_id": "", "online_name": "", "short_description": "", "long_description": "",
    "ingredients": [], "directions": [], "warnings": [], "features": [],
    "specifications": {"size": "", "flavour": "", "shade": "", "dimensions": ""},
    "categories": [], "variants": [],
}
PRODUCTS = [
    ("health", "Cape Care", "Vitamin C Tablets", "60 g", "Orange", "Ascorbic acid, sucrose", "Take one tablet daily", "Keep out of reach of children", "Sugar coated"),
    ("baby", "Little Sprout", "Baby Wash", "250 ml", "", "Aqua, glycerin", "Apply to wet skin and rinse", "For external use only", "Fragrance free"),
    ("beauty", "Veld Bloom", "Lip Balm", "5 g", "", "Beeswax, shea butter", "Apply to lips as needed", "Avoid contact with eyes", "Moisturising"),
    ("food", "Karoo Pantry", "Oat Porridge", "500 g", "Vanilla", "Oats, vanilla", "Mix with hot water", "Contains oats", "High in fibre"),
    ("household", "Home Kind", "Dishwashing Liquid", "750 ml", "Lemon", "Water, surfactants", "Add a small amount to warm water", "Do not swallow", "Cuts grease"),
]


def make_synthetic(out, seed=42, count=60):
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    schema = Schema(ASSUMED_FIELDS)
    (root / "submission_format.json").write_text(json.dumps(ASSUMED_FIELDS, indent=2) + "\n")
    rows, truth = [], []
    for i in range(count + 14):
        category, brand, product, size, flavour, ingredients, usage, warning, feature = rng.choice(PRODUCTS)
        name = f"{brand} {product}{' ' + flavour if flavour else ''} {size}"
        summary = f"{product} for everyday use"
        description = f"{brand} {product}. {feature}. {size}."
        mode = i % 9
        row = {"record_id": f"{i:07d}", "brnd": brand, "prod_nm": name,
               "supplier_copy": "", "web_copy": "", "attribute_notes": "",
               "category": category, "do_not_predict_fields": "[]"}
        reference = {field: schema.empty(field) for field in schema.columns}
        reference.update(article_id=row["record_id"], online_name=name, short_description=summary,
                         long_description=description, ingredients=ingredients.split(", "), directions=[usage],
                         warnings=[warning], features=[feature], categories=[category])
        reference["specifications"].update(size=size, flavour=flavour)
        # Several grammars deliberately are not all supported by the baseline.
        if mode in (0, 1, 5, 8):
            row["supplier_copy"] = f"Ingredients: {ingredients}; Usage: {usage}; Warning: {warning}; Benefits: {feature}"
            row["web_copy"] = f"Summary: {summary}; Description: {description}"
        elif mode in (2, 6):
            row["supplier_copy"] = f"Made with {ingredients}. To use, {usage.lower()}. Please note: {warning.lower()}."
            row["web_copy"] = f"{summary}. {description}"
        else:
            row["supplier_copy"] = f"Composition: {ingredients}\nHow to use: {usage}\nCaution: {warning}\nClaims: {feature}"
            row["web_copy"] = f"Short description: {summary}\nLong description: {description}"
        row["attribute_notes"] = f"Pack size: {size}" + (f"; Flavour: {flavour}" if flavour else "")
        if category == "beauty":
            row["attribute_notes"] += "; Shade: Clear"
            reference["specifications"]["shade"] = "Clear"
        if mode == 1:
            row["web_copy"] += "; Ingredients: Water, salt"
            reference["ingredients"] = []  # Contradictory sources: blank is correct.
        if mode == 3:
            row["attribute_notes"] += "; Size: 100 ml"
            reference["specifications"]["size"] = ""
            reference["online_name"] = ""
        if mode == 4:
            row["supplier_copy"] = "[MASKED]"
            for field in ("ingredients", "directions", "warnings", "features"):
                reference[field] = []
        if mode == 5:
            row["do_not_predict_fields"] = '["warnings", "variants"]'
            reference["warnings"] = []
        if mode == 6:
            row["prod_nm"] = "[MASKED]"
            # Identity cannot be reconstructed confidently from this phrasing.
            reference["online_name"] = ""
        if mode == 7:
            row["supplier_copy"] = ""
            row["web_copy"] = "[MASKED]"
            for field in ("ingredients", "directions", "warnings", "features"):
                reference[field] = []
            reference["short_description"] = reference["long_description"] = ""
        if mode == 8:
            row["attribute_notes"] += "; Variant: Travel pack"
            reference["variants"] = ["Travel pack"]
        rows.append(row)
        truth.append(reference)
    columns = list(rows[0]) if rows else []
    write_csv(root / "Test_Masked.csv", columns, rows[:count])
    write_csv(root / "template_input.csv", columns, rows[count:])
    write_submission(root / "ground_truth.csv", truth[:count], schema)
    write_submission(root / "template.csv", truth[count:], schema)
    write_submission(root / "SampleSubmission.csv", [{**{f: schema.empty(f) for f in schema.columns}, "article_id": r["record_id"]} for r in rows[:count]], schema)
    # Different identifiers, never a key-based answer source for the held-out set.
    catalogue = []
    for i in range(len(rows)):
        cat, brand, product, size, flavour, ingredients, usage, warning, feature = PRODUCTS[i % len(PRODUCTS)]
        catalogue.append({"record_id": f"U{i:07d}", "brnd": brand, "prod_nm": f"{brand} {product} {size}",
                          "supplier_copy": f"Ingredients: {ingredients}; Usage: {usage}; Warning: {warning}",
                          "web_copy": f"Benefits: {feature}", "attribute_notes": f"Size: {size}",
                          "category": cat, "do_not_predict_fields": "[]"})
    write_csv(root / "unlabelled_data.csv", columns, catalogue)
    (root / "ASSUMPTIONS.md").write_text("SYNTHETIC ONLY. The seven structured names ingredients, directions, warnings, features, specifications, categories, variants are ASSUMED; not confirmed Zindi fields. Ground truth is held out from inference.\n")
    return root
