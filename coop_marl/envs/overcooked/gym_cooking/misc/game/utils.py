import pygame


class Color:
    BLACK = (0, 0, 0)
    FLOOR = (245, 230, 210)  # light gray
    COUNTER = (220, 170, 110)  # tan/gray
    COUNTER_BORDER = (114, 93, 51)  # darker tan
    DELIVERY = (96, 96, 96)  # grey


KeyToTuple = {
    pygame.K_UP: (0, -1),  # 273
    pygame.K_DOWN: (0, 1),  # 274
    pygame.K_RIGHT: (1, 0),  # 275
    pygame.K_LEFT: (-1, 0),  # 276
}

KeyToTuple_human1 = {
    pygame.K_UP: 4,  # 273
    pygame.K_DOWN: 3,  # 274
    pygame.K_RIGHT: 2,  # 275
    pygame.K_LEFT: 1,  # 276
    pygame.K_SPACE: 0,
    pygame.K_f: 5
}

KeyToTuple_human2 = {
    pygame.K_w: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_d: (1, 0),
    pygame.K_a: (-1, 0),
}
