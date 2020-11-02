import os


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'model')

if not os.path.exists(MODELS_ROOT):
    os.mkdir(MODELS_ROOT)
