""" Constants for the DSLR project. """

HOUSES = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']

HOUSE_COLORS = {
    'Gryffindor': 'red',
    'Ravenclaw': 'blue',
    'Hufflepuff': 'green',
    'Slytherin': 'yellow',
}

COURSES = [
    'Arithmancy', 'Astronomy', 'Herbology',
    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
    'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
    'Care of Magical Creatures', 'Charms', 'Flying'
]

FEATURES_TO_DROP = ['Care of Magical Creatures', 'Arithmancy', 'Defense Against the Dark Arts', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']

IMAGES_FOLDER = 'images'

MODEL_SETUPS = {
    'learning_rate': 0.1,
    'epochs': 10000,
}
