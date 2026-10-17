"""Recipes: functions seed -> SceneSpec that generate whole families of scenes (and the reference data for training)."""
from .warehouse import warehouse

RECIPES = {"warehouse": warehouse}


DESCRIPTIONS = {
    "warehouse": "40 x 25 m fulfillment warehouse: rows of pallet racks loaded with boxes, dock doors with levelers, belt and roller conveyors "
                 "feeding a packing table, a caged area, charging stations, a forklift loop, two walking people, floor markings, clutter "
                 "and spilled cartons, a 0.9 m chokepoint, and a robot with pick / dock / chokepoint goals.",
}


def list_recipes() -> list[str]:
    return sorted(RECIPES)


def get_recipe(name: str):
    return RECIPES[name]
