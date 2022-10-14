from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_book.recipe import Recipe, RecipeNode
from copy import deepcopy


def id_num_generator():
    num = 0
    while True:
        yield num
        num += 1


id_generator = id_num_generator()

#  Basic food Items
# root_type, id_num, parent=None, conditions=None, contains=None
ChoppedLettuce = RecipeNode(root_type=Lettuce, id_num=next(id_generator), name="Lettuce",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedOnion = RecipeNode(root_type=Onion, id_num=next(id_generator), name="Onion",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedTomato = RecipeNode(root_type=Tomato, id_num=next(id_generator), name="Tomato",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)])
MashedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                          conditions=[("blend_state", BlenderFoodStates.MASHED)])

# Salad Plates
LettuceSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedLettuce])
TomatoSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedTomato])
TomatoLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedTomato, ChoppedLettuce])
TomatoCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedTomato, ChoppedCarrot])
TomatoLettuceOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                     contains=[ChoppedTomato, ChoppedLettuce, ChoppedOnion])
ChoppedOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedOnion])
ChoppedCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                         contains=[ChoppedCarrot])
MasedCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                         contains=[MashedCarrot])

# Delivered Salads
LettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare", conditions=None,
                         contains=[LettuceSaladPlate])
TomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare", conditions=None,
                         contains=[TomatoSaladPlate])
TomatoLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[TomatoLettucePlate])
TomatoCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[TomatoCarrotPlate])
ChoppedOnion = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[ChoppedOnionPlate])
ChoppedCarrot = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[ChoppedCarrotPlate])
MashedCarrot = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[MasedCarrotPlate])

# this one increments one further and is thus the amount of ids we have given since
# we started counting at zero.
NUM_GOALS = next(id_generator)

RECIPES = {
            "LettuceSalad":lambda: deepcopy(Recipe(LettuceSalad, name='LettuceSalad')),
            "TomatoSalad": lambda: deepcopy(Recipe(TomatoSalad, name='TomatoSalad')),
            "TomatoLettuceSalad": lambda: deepcopy(Recipe(TomatoLettuceSalad, name='TomatoLettuceSalad')),
            "TomatoCarrotSalad": lambda: deepcopy(Recipe(TomatoCarrotSalad, name='TomatoCarrotSalad')),
            # "TomatoLettuceOnionSalad": lambda: deepcopy(Recipe(TomatoLettuceOnionSalad, name='TomatoLettuceOnionSalad')),
            "ChoppedCarrot": lambda: deepcopy(Recipe(ChoppedCarrot, name='ChoppedCarrot')), 
            "ChoppedOnion": lambda: deepcopy(Recipe(ChoppedOnion, name='ChoppedOnion')), 
            "MashedCarrot": lambda: deepcopy(Recipe(MashedCarrot, name='MashedCarrot'))}
