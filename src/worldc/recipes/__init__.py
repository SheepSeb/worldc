"""Recipes: functions seed -> SceneSpec that generate whole families of scenes (and the reference data for training)."""
from .warehouse import warehouse

RECIPES = {"warehouse": warehouse}


def list_recipes() -> list[str]:
    return sorted(RECIPES)


def get_recipe(name: str):
    return RECIPES[name]
